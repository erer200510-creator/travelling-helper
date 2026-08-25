"""
知识库构建脚本

用法:
  python scripts/build_kb.py

执行内容(对应简历'利用 Python 脚本完成 JSON 数据清洗格式化,构建高质量向量数据库'):
  1. 读取 data/travel_cases.json
  2. 清洗归一化(loader)
  3. 中文感知切片(chunker)
  4. Embedding 向量化(embeddings)
  5. 写入向量数据库(vector_store)并持久化
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

import config  # noqa: E402
from app.rag.chunker import chunk_documents  # noqa: E402
from app.rag.embeddings import get_embeddings  # noqa: E402
from app.rag.loader import load_kb_documents  # noqa: E402
from app.rag.vector_store import create_vector_store  # noqa: E402


def main() -> None:
    print(f"① 加载知识库文件: {config.KB_FILE}")
    raw_docs = load_kb_documents()
    print(f"   清洗后语义文档: {len(raw_docs)} 条")

    print("② 中文切片(Chunking)...")
    chunks = chunk_documents(raw_docs)
    print(f"   chunk 总数: {len(chunks)} 条")

    print("③ Embedding 向量化...")
    emb = get_embeddings()
    print(f"   provider: {emb.name} (dim={emb.dim})")

    print("④ 写入向量数据库...")
    store = create_vector_store(embedder=emb)
    store.clear()
    store.add_documents(chunks)
    print(f"   已写入 {len(store)} 条向量,持久化目录: {config.VECTOR_DIR}")

    # 自检:检索一次,打印 top3
    print("⑤ 自检检索: '西安 兵马俑 门票 半天'")
    hits = store.similarity_search("西安 兵马俑 门票 半天", k=3, score_threshold=0.0)
    for h in hits:
        print(f"   - [{h.metadata.get('score'):.3f}] {h.page_content[:70].replace(chr(10),' ')}")
    print("知识库构建完成 ✔")


if __name__ == "__main__":
    main()
