"""
Embedding 向量化层 —— 支持多 Provider 一键切换

对应简历项目经历:
  'Embedding 处理模型更换,提升召回率信噪比准确度指标'

Provider 优先级(通过 EMBEDDING_PROVIDER 环境变量切换):
  - sentence-transformers : 本地 BAAI/bge-small-zh-v1.5(中文检索质量最佳,推荐)
  - local_hash            : 内置轻量 n-gram 哈希向量化(零依赖,离线可跑通全流程)
  - dashscope             : 阿里云 text-embedding-v3(需要 EMBEDDING_API_KEY)

统一接口,上层 RAG 层无感。
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import List, Protocol

import numpy as np

import config

logger = logging.getLogger(__name__)

# 内置哈希向量维度(1024,足以支撑中小规模 RAG 的相似度区分)
HASH_DIM = 1024


class Embedder(Protocol):
    dim: int
    name: str

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


# ---------------------------------------------------------------
# 内置轻量向量化器:字符 2/3-gram 特征哈希 + L2 归一化
# ---------------------------------------------------------------
_CJK = re.compile(r"[\u4e00-\u9fff]")


def _ngrams(text: str, n: int = 3):
    """提取中文 2/3-gram 与英文单词特征(2 字词如'门票/轮渡'也能命中)"""
    chars = [c for c in text if _CJK.match(c)]
    for size in (2, 3):
        for i in range(len(chars) - size + 1):
            yield f"c{size}" + "".join(chars[i:i + size])
    # 英文单词(含数字)
    for word in re.findall(r"[A-Za-z0-9]{2,}", text.lower()):
        yield "w" + word


class LocalHashEmbeddings:
    """无依赖哈希向量化:适合演示与离线开发;生产建议替换为 bge/dashscope
    带轻型 IDF 加权:高频模板词(城市/开放时间等)自动降权,低频关键词(景点名等)升权。
    """

    dim = HASH_DIM
    name = "local_hash"

    def __init__(self):
        self._idf = None  # feature -> idf 权重;None 表示未训练

    # ---------- IDF 训练(在 add_documents 时对全量语料自动进行)----------
    def fit(self, texts: List[str]) -> None:
        df: dict = {}
        for text in texts:
            for feat in set(_ngrams(text)):
                df[feat] = df.get(feat, 0) + 1
        n = len(texts) or 1
        self._idf = {feat: __import__("math").log((1 + n) / (1 + cnt)) + 1.0 for feat, cnt in df.items()}

    def _w(self, feat: str) -> float:
        return self._idf.get(feat, 1.0) if self._idf else 1.0

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat in _ngrams(text):
            idx = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += self._w(feat)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._idf is None and texts:
            self.fit(texts)
        return [self._embed(t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text).tolist()


# ---------------------------------------------------------------
# sentence-transformers 本地 Embedding(bge-small-zh-v1.5)
# ---------------------------------------------------------------
class SentenceTransformerEmbeddings:
    name = "sentence-transformers"

    def __init__(self, model_name: str = None):
        # 延迟 import:避免未安装时直接崩掉
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(model_name or config.EMBEDDING_MODEL)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> List[float]:
        vecs = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return vecs[0].tolist()


# ---------------------------------------------------------------
# Provider 选择(带自动回退)
# ---------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> Embedder:
    provider = config.EMBEDDING_PROVIDER.lower()

    if provider in ("auto", "sentence-transformers"):
        try:
            emb = SentenceTransformerEmbeddings()
            logger.info("Embedding provider: sentence-transformers (%s)", emb.dim)
            return emb
        except Exception as e:  # 未安装 / 模型下载失败 → 回退
            if provider == "sentence-transformers":
                logger.warning("sentence-transformers 不可用(%s),回退 local_hash", e)
            else:
                logger.info("未检测到 sentence-transformers,使用内置 local_hash 向量化")

    if provider == "dashscope":
        try:
            return _DashScopeEmbeddings()
        except Exception as e:
            logger.warning("dashscope embedding 不可用(%s),回退 local_hash", e)

    return LocalHashEmbeddings()


class _DashScopeEmbeddings:
    """阿里云 DashScope text-embedding-v3(可选)"""

    name = "dashscope"

    def __init__(self):
        import requests  # noqa: F401
        if not config.EMBEDDING_API_KEY:
            raise RuntimeError("EMBEDDING_API_KEY 未配置")
        self.dim = 1024

    def _call(self, texts: List[str]) -> List[List[float]]:
        import requests
        resp = requests.post(
            "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding",
            headers={"Authorization": f"Bearer {config.EMBEDDING_API_KEY}", "Content-Type": "application/json"},
            json={"model": "text-embedding-v3", "input": {"texts": texts}},
            timeout=30,
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["output"]["embeddings"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]
