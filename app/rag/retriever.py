"""
检索增强工具(Retriever)

对应简历项目经历:
  'RAG 检索 → 可行性评估 → 旅行建议生成' 流程中的核心检索组件

设计:
  - 双路召回:语义向量召回 + 元数据(城市)软过滤
  - 城市/季节/天数元数据过滤保证命中同城案例优先
  - 检索结果格式化为带来源锚点的上下文,供 LLM 归因引用
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

import config
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import create_vector_store

logger = logging.getLogger(__name__)

# 各文档类型的配额:保证行程生成所需知识(景点/美食/住宿/预算/可行性/交通)都齐全
_TYPE_QUOTA = {
    "overview": 1,
    "attraction": 4,
    "food": 1,
    "hotel": 1,
    "budget": 1,
    "feasibility": 1,
    "transport": 1,
}

def _diverse_pick(docs: List[Document], k: int, type_priority=None) -> List[Document]:
    """
    两阶段类型均衡选取 top-k:
      阶段1:按类型优先级,每类先保证 top-1(行程用到的知识不能缺类);
      阶段2:剩余名额按相似度补足(景点类最多 TYPE_QUOTA['attraction'] 条)。
    """
    if len(docs) <= k:
        return docs
    priority = list(type_priority or ("overview", "attraction", "food", "hotel", "budget", "feasibility", "transport"))
    by_type: dict = {}
    for d in docs:
        by_type.setdefault(d.metadata.get("doc_type", "overview"), []).append(d)
    for lst in by_type.values():
        lst.sort(key=lambda x: -float(x.metadata.get("score", 0)))

    picked: List[Document] = []
    picked_ids = set()

    def _pick(d: Document) -> bool:
        nonlocal picked, picked_ids
        if len(picked) >= k or id(d) in picked_ids:
            return False
        picked.append(d)
        picked_ids.add(id(d))
        return True

    # 阶段 1:每类先取一条
    for t in priority:
        if len(picked) >= k:
            break
        if by_type.get(t):
            _pick(by_type[t][0])
    # 阶段 2:按分数补足(受配额限制:attraction ≤ 4,其余 ≤ 1)
    for d in sorted(docs, key=lambda x: -float(x.metadata.get("score", 0))):
        if len(picked) >= k:
            break
        t = d.metadata.get("doc_type", "overview")
        used = sum(1 for p in picked if p.metadata.get("doc_type") == t)
        if used < _TYPE_QUOTA.get(t, 1):
            _pick(d)
    return picked


class TravelRetriever:
    def __init__(self, store=None, embedder=None, top_k: int = None, score_threshold: float = None):
        self.embedder = embedder or get_embeddings()
        self.store = store or create_vector_store(embedder=self.embedder)
        self.top_k = top_k or config.RAG_TOP_K
        self.score_threshold = score_threshold if score_threshold is not None else config.RAG_SCORE_THRESHOLD

    def __len__(self) -> int:
        return len(self.store)

    # ------------- 检索 -------------
    def search(self, query: str, k: int | None = None,
               city: str = "", metadata_filter: Dict[str, Any] | None = None) -> List[Document]:
        """
        语义检索,返回按相似度降序的 Document 列表(已带 score)。

        策略:
          - 城市已知:同城文档优先(按相似度降序),同城不足 k 条时用全网补齐;
            此时不做全局阈值截断,保证行程生成所需的多类知识(景点/美食/住宿/预算)齐全。
          - 城市未知:自适应阈值 min(绝对阈值, top1*0.6),兼顾
            bge(0.4+ 分布)与 local_hash(0.1-0.3 分布)两类向量空间。
        """
        k = k or self.top_k
        if city:
            # 城市已知:用元数据过滤获取全部同城文档(不受全局 top-k 干扰),
            # 再按分数做类型均衡选取;同城不足 k 条时用全网高分补齐。
            same_city = self.store.similarity_search(
                query, k=50, score_threshold=0.0, metadata_filter={"city": city},
            )
            docs = _diverse_pick(same_city, k)
            if len(docs) < k:
                others = self.store.similarity_search(query, k=k * 2, score_threshold=0.0)
                others = [d for d in others if d.metadata.get("city") != city]
                docs += others[: k - len(docs)]
            return docs

        docs = self.store.similarity_search(query, k=k * 4, score_threshold=0.0)
        # 自适应阈值截断:min(绝对阈值, top1*0.6)
        if docs:
            top = float(docs[0].metadata.get("score", 0.0))
            cutoff = min(self.score_threshold, top * 0.6)
            docs = [d for d in docs if float(d.metadata.get("score", 0.0)) >= cutoff]
        return docs[:k]

    def search_by_city(self, city: str, k: int = 3) -> List[Document]:
        """按城市直接带过滤检索(用于可行性评估节点)"""
        return self.store.similarity_search(
            query=f"{city} 旅游 景点 美食 住宿 预算",
            k=k,
            score_threshold=0.0,
            metadata_filter={"city": city},
        )

    def search_city_section(self, city: str, doc_type: str, k: int = 6) -> List[Document]:
        """按城市 + 文档类型检索(如补全同城全部景点,供行程生成使用)"""
        return self.store.similarity_search(
            query=f"{city} {doc_type}",
            k=k,
            score_threshold=0.0,
            metadata_filter={"city": city, "doc_type": doc_type},
        )

    # ------------- 格式化 -------------
    @staticmethod
    def format_context(docs: List[Document], max_per_doc: int = 1200) -> str:
        """检索结果 → prompt 上下文(带来源标注)"""
        blocks = []
        for i, d in enumerate(docs, 1):
            content = d.page_content[:max_per_doc]
            meta = d.metadata or {}
            tag = f"[知识库#{i} | {meta.get('city', '')} | {meta.get('doc_type', '')} | 相关度 {meta.get('score', '-'):.3f}]" if isinstance(meta.get("score"), float) else f"[知识库#{i} | {meta.get('city', '')} | {meta.get('doc_type', '')}]"
            blocks.append(f"{tag}\n{content}")
        return "\n\n".join(blocks)


def build_retriever() -> TravelRetriever:
    """构建(或从磁盘加载)检索器"""
    return TravelRetriever()


def ensure_knowledge_base(docs=None) -> TravelRetriever:
    """
    确保知识库可用:空库时自动从 data/travel_cases.json 构建。
    构建脚本为 scripts/build_kb.py(独立执行亦可)。
    """
    retriever = build_retriever()
    if len(retriever) == 0:
        from app.rag.loader import load_kb_documents
        from app.rag.chunker import chunk_documents
        raw_docs = docs or chunk_documents(load_kb_documents())
        retriever.store.add_documents(raw_docs)
        logger.info("知识库构建完成: 共 %d 条 chunk", len(raw_docs))
    return retriever
