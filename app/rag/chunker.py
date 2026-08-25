"""
中文文本切片(Chunking)优化层

对应简历项目经历:
  '向量数据库 Chunking 切片优化,提升召回率信噪比准确度指标'

设计要点:
  1. 文档已按语义切分(总览/景点/美食/住宿/预算/可行性),这里只对超长文档按
     中文字符粒度二次切分,避免把不同语义内容混进同一 chunk(信噪比指标)。
  2. 使用中文感知分隔符优先级:换行 > 句号 > 分号 > 逗号,保留 20% 重叠。
  3. 切分前剥离标题与正文,保证每个 chunk 自带来源锚点,便于归因。
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# 中文感知分隔符:按语义边界从大到小
_CHINESE_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", " ", ""]


def get_splitter(chunk_size: int | None = None, overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=overlap or config.CHUNK_OVERLAP,
        separators=_CHINESE_SEPARATORS,
        # 中英文混排按字符数近似:500 tokens ≈ 350 汉字左右
        length_function=len,
        keep_separator=True,
    )


def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    语义文档 → 入库 chunk。

    规则:
      - 长度 < chunk_size 的文档原样入库(保留文档级语义,召回更准)
      - 超长文档用分隔符切分;切分后每个 chunk 保留原 metadata,
        并在 content 首行追加来源标题锚点。
    """
    splitter = get_splitter()
    result: List[Document] = []
    seen: set = set()

    for doc in docs:
        # 提取首行标题作为锚点(如 【景点】故宫博物院(北京))
        first_line = doc.page_content.split("\n", 1)[0].strip()
        if len(doc.page_content) <= config.CHUNK_SIZE:
            chunks = [doc]
        else:
            chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if len(content) < 10:
                continue
            # 非首个 chunk 补锚点,让模型知道这段内容归属
            if idx > 0 and first_line not in content:
                content = f"{first_line}\n{content}"
            key = (content[:80], tuple(sorted((chunk.metadata or {}).items())))
            if key in seen:
                continue
            seen.add(key)
            chunk.page_content = content
            chunk.metadata = {**(chunk.metadata or {}), "chunk_id": idx, "anchor": first_line}
            result.append(chunk)
    return result
