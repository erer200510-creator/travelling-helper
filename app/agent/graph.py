"""
LangGraph 工作流编排(Agent 主体)

对应简历项目经历:
  '助手流程:用户输入/语音识别 → LLM 理解 → RAG 检索 → 可行性评估 → 旅行建议生成'

图结构:
  START → parse → (信息不足?) → clarify → END
                 ↓(信息足够)
               retrieve → evaluate → generate → END
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (clarify_node, evaluate_node, generate_node,
                             parse_node, retrieve_node, route_after_parse)
from app.agent.state import TravelState

logger = logging.getLogger(__name__)

_graph = None


def build_graph():
    """构建并编译工作流图(单例)"""
    global _graph
    if _graph is not None:
        return _graph

    graph = StateGraph(TravelState)
    graph.add_node("parse", parse_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", route_after_parse, {"clarify": "clarify", "retrieve": "retrieve"})
    graph.add_edge("clarify", END)
    graph.add_edge("retrieve", "evaluate")
    graph.add_edge("evaluate", "generate")
    graph.add_edge("generate", END)

    _graph = graph.compile()
    return _graph


def run_agent(user_request: str, verbose: bool = False,
              prev_plan: str = "", is_followup: bool = False,
              history: list | None = None) -> Dict[str, Any]:
    """
    运行 Agent,返回完整执行轨迹:
      {
        'request', 'parsed', 'missing_fields', 'context_text', 'retrieved_docs',
        'evaluation', 'weather_data', 'plan' 或 'reply'
      }
    prev_plan: 多轮对话中上一版行程(非空时按修订模式生成新版行程)
    history:   对话历史(跨轮上下文记忆)
    """
    graph = build_graph()
    initial: TravelState = {
        "user_request": user_request,
        "prev_plan": prev_plan,
        "is_followup": is_followup,
        "history": history or [],
    }
    result = graph.invoke(initial)
    if verbose:
        for step, updates in result.get("__intermediate__", {}).items():
            logger.info("STEP %s: %s", step, list(updates.keys()))
    return result


def stream_agent(user_request: str, prev_plan: str = "", is_followup: bool = False,
                 history: list | None = None):
    """逐节点流式执行(供 CLI/UI 展示每一步)

    history: 对话历史 [{"role": "user"/"assistant", "content": str, "is_plan": bool}],
             用于跨轮上下文记忆(目的地/偏好/预算等不再重复询问)。
    """
    graph = build_graph()
    for update in graph.stream(
        {
            "user_request": user_request,
            "prev_plan": prev_plan,
            "is_followup": is_followup,
            "history": history or [],
        },
        stream_mode="updates",
    ):
        yield update
