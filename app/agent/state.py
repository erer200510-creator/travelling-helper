"""
LangGraph 工作流状态定义

工作流(与简历项目流程一一对应):
  用户输入/语音识别 → LLM 理解(parse) → RAG 检索(retrieve)
    → 可行性评估(evaluate) → 旅行建议生成(generate)
  不足信息时走澄清分支(clarify)
"""
from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class TravelState(TypedDict, total=False):
    # 输入
    user_request: str
    prev_plan: str                # 上一版行程(多轮对话修订场景)
    is_followup: bool             # 是否基于上一版行程的追问/修改
    history: List[Dict[str, str]] # 对话历史 [{"role": "user"/"assistant", "content": str, "is_plan": bool}]

    # 节点 1:LLM 理解
    parsed: Dict[str, Any]            # 需求结构化结果
    missing_fields: List[str]         # 缺失的关键字段

    # 节点 2:RAG 检索
    retrieved_docs: List[Dict[str, Any]]  # 文档摘要(含 score/metadata)
    context_text: str                 # 已格式化的知识库上下文

    # 节点 3:可行性评估
    evaluation: Dict[str, Any]
    weather_data: str                 # 天气工具原始输出
    tool_data: str                    # 工具数据汇总(天气/高德)

    # 节点 4:行程生成
    plan: str                         # 最终行程 Markdown

    # 澄清分支
    reply: str

    # 兜底
    error: str
