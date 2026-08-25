"""
向量数据库层

对应简历项目经历:
  '构建 RAG 知识库系统 / 构建高质量向量数据库'

实现:
  - SimpleVectorStore:内置轻量向量库(numpy + 余弦相似度,JSON 持久化),零额外依赖
  - ChromaVectorStore :可选的 ChromaDB 向量库(安装 chromadb 后自动启用)

两者实现同一接口,上层无感切换。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from langchain_core.documents import Document

import config

logger = logging.getLogger(__name__)


def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query_vec(归一化) × 文档矩阵(按行归一化)"""
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    return matrix @ query_vec


class SimpleVectorStore:
    """
    内置轻量向量库:余弦相似度 + 持久化到 storage/vector_index/*.npz+*.json
    适合中小规模知识库(千级文档)与离线演示。
    """

    def __init__(self, persist_dir: Path | None = None, embedder=None):
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.embedder = embedder
        self._docs: List[Document] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim) 已归一化

    # ---------- 写入 ----------
    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            return
        vecs = np.asarray(self.embedder.embed_documents([d.page_content for d in docs]), dtype=np.float32)
        vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        self._docs.extend(docs)
        if self._matrix is None:
            self._matrix = vecs
        else:
            self._matrix = np.vstack([self._matrix, vecs])
        if self.persist_dir:
            self._persist()

    # ---------- 检索 ----------
    def similarity_search(self, query: str, k: int = 4, score_threshold: float | None = None,
                          metadata_filter: Dict[str, Any] | None = None) -> List[Document]:
        if not self._docs:
            return []
        q = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = _cosine_scores(q, self._matrix)

        idxs = np.argsort(-scores)
        results: List[Document] = []
        for i in idxs:
            doc = self._docs[i]
            if score_threshold is not None and float(scores[i]) < score_threshold:
                continue
            if metadata_filter:
                ok = all(doc.metadata.get(k) in (v if isinstance(v, list) else [v]) for k, v in metadata_filter.items())
                if not ok:
                    continue
            copied = Document(page_content=doc.page_content, metadata={**(doc.metadata or {}), "score": round(float(scores[i]), 4)})
            results.append(copied)
            if len(results) >= k:
                break
        return results

    def __len__(self) -> int:
        return len(self._docs)

    # ---------- 持久化 ----------
    def _persist(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.persist_dir / "vectors.npz", matrix=self._matrix)
        payload = [
            {"content": d.page_content, "metadata": d.metadata}
            for d in self._docs
        ]
        (self.persist_dir / "docs.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        # 持久化局部哈希向量器的 IDF 权重,保证加载后查询与入库权重一致
        idf = getattr(self.embedder, "_idf", None) if self.embedder else None
        if idf:
            (self.persist_dir / "idf.json").write_text(json.dumps(idf, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, persist_dir: Path, embedder) -> "SimpleVectorStore":
        store = cls(persist_dir=persist_dir, embedder=embedder)
        npz = persist_dir / "vectors.npz"
        docs_json = persist_dir / "docs.json"
        if npz.exists() and docs_json.exists():
            data = np.load(npz)
            store._matrix = data["matrix"]
            payload = json.loads(docs_json.read_text(encoding="utf-8"))
            store._docs = [Document(page_content=p["content"], metadata=p["metadata"]) for p in payload]
            idf_file = persist_dir / "idf.json"
            if idf_file.exists() and hasattr(embedder, "_idf"):
                embedder._idf = json.loads(idf_file.read_text(encoding="utf-8"))
            logger.info("已加载内置向量库: %d 条文档", len(store._docs))
        return store

    def clear(self) -> None:
        self._docs, self._matrix = [], None
        if self.persist_dir and self.persist_dir.exists():
            for f in self.persist_dir.iterdir():
                f.unlink()


# ---------------------------------------------------------------
# ChromaDB 可选实现
# ---------------------------------------------------------------
class ChromaVectorStore:
    """ChromaDB 向量库(chromadb 未安装时不可用,由 build_kb 自动抉择)"""

    def __init__(self, persist_dir: Path, embedder):
        import chromadb  # type: ignore
        self.embedder = embedder
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(
            name="travel_kb",
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: List[Document]) -> None:
        texts = [d.page_content for d in docs]
        vecs = self.embedder.embed_documents(texts)
        ids = [f"doc_{i}" for i in range(len(texts))]
        metadatas = [
            {k: (str(v) if not isinstance(v, (int, float, bool, str)) else v) for k, v in d.metadata.items()}
            for d in docs
        ]
        self._col.upsert(ids=ids, embeddings=vecs, documents=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4, score_threshold: float | None = None,
                          metadata_filter: Dict[str, Any] | None = None) -> List[Document]:
        q = self.embedder.embed_query(query)
        kwargs: Dict[str, Any] = {"query_embeddings": [q], "n_results": k}
        if metadata_filter:
            kwargs["where"] = {"$and": [{kk: vv} for kk, vv in metadata_filter.items()]}
        res = self._col.query(**kwargs)
        out = []
        for content, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            # cosine distance → similarity
            score = 1.0 - dist
            if score_threshold is not None and score < score_threshold:
                continue
            m = {k: v for k, v in meta.items() if v is not None}
            m["score"] = round(float(score), 4)
            out.append(Document(page_content=content, metadata=m))
        return out

    def __len__(self) -> int:
        return self._col.count()


def create_vector_store(persist_dir: Path | None = None, embedder=None):
    """
    工厂方法:优先 ChromaDB,回退内置 SimpleVectorStore。
    已有持久化数据时直接加载。
    """
    persist_dir = persist_dir or config.VECTOR_DIR
    try:
        store = ChromaVectorStore(persist_dir, embedder)
        logger.info("向量库: ChromaDB (%d docs)", len(store))
        return store
    except ImportError:
        try:
            return SimpleVectorStore.load(persist_dir, embedder)
        except Exception:
            return SimpleVectorStore(persist_dir=persist_dir, embedder=embedder)
    except Exception as e:
        logger.warning("ChromaDB 初始化失败(%s),回退内置向量库", e)
        try:
            return SimpleVectorStore.load(persist_dir, embedder)
        except Exception:
            return SimpleVectorStore(persist_dir=persist_dir, embedder=embedder)
