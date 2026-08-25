"""
LangGraph 工作流节点实现

流程(与简历项目一致):
  parse(LLM 理解) → retrieve(RAG 检索) → evaluate(可行性评估) → generate(旅行建议生成)
  parse 信息不足 → clarify(澄清追问)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

import config
from app.agent.fallback_plan import build_template_plan
from app.agent.state import TravelState
from app.llm import get_llm, llm_is_mock, llm_json, run_tool_calls
from app.prompts import (CLARIFY_SYSTEM, EVALUATE_SYSTEM, PARSE_SYSTEM,
                         PLAN_SYSTEM, PLAN_USER_TEMPLATE)
from app.rag.retriever import TravelRetriever, build_retriever
from app.tools.amap import get_amap_tools
from app.tools.budget import estimate_budget, format_budget
from app.tools.weather import get_weather_forecast

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# 对话历史格式化:把跨轮对话压成一段紧凑文本注入提示词
# ---------------------------------------------------------------
HISTORY_MAX_MESSAGES = 8      # 最多携带最近 N 条消息
HISTORY_MAX_CHARS = 2800      # 历史总长上限(字符)
HISTORY_MSG_CHARS = 320       # 单条消息截断


def format_history(history: list | None) -> str:
    """[{"role": "user"/"assistant", "content": str, "is_plan": bool}] → 文本段"""
    if not history:
        return "无"
    lines = []
    for m in history[-HISTORY_MAX_MESSAGES:]:
        role = "用户" if m.get("role") == "user" else "助手"
        content = str(m.get("content", "") or "").replace("\n", " ")
        if m.get("is_plan"):
            content = f"[行程] {content[:HISTORY_MSG_CHARS]}"
        else:
            content = content[:HISTORY_MSG_CHARS]
        if content.strip():
            lines.append(f"- {role}: {content}")
    text = "\n".join(lines)
    return text[:HISTORY_MAX_CHARS] or "无"


# ---------------------------------------------------------------
# 节点 1:LLM 理解(需求解析)
# ---------------------------------------------------------------
class TravelRequest(BaseModel):
    destination: str = Field(default="", description="目的地城市/景点")
    start_date: str = Field(default="", description="出发日期 YYYY-MM-DD")
    days: int = Field(default=0, description="行程天数")
    budget_total: int = Field(default=0, description="人均总预算(元)")
    people: int = Field(default=1, description="出行人数")
    preferences: List[str] = Field(default_factory=list, description="偏好标签列表")
    transport_mode: str = Field(default="", description="出发交通方式")
    notes: str = Field(default="", description="其他需求备注")


def parse_node(state: TravelState) -> Dict[str, Any]:
    req_text = state["user_request"]
    # 历史注入系统提示词:真实 LLM 可理解省略说法(如"预算降到2000"),Mock 演示仅解析本轮【用户】段
    sys_prompt = PARSE_SYSTEM
    if state.get("history"):
        sys_prompt += (
            "\n\n【对话历史(较早轮次,作为补充背景,用于理解省略说法)】\n"
            + format_history(state["history"])
        )
    parsed = llm_json(sys_prompt, req_text, TravelRequest)
    if parsed is None:
        logger.warning("需求解析失败,走澄清分支")
        return {"parsed": {}, "missing_fields": ["destination", "days"]}

    data = parsed.model_dump()
    # 规范天数:0 → 默认 3
    if not data["days"] or data["days"] < 1:
        data["days"] = 3
    missing = []
    if not data["destination"]:
        missing.append("destination")
    if not data["start_date"]:
        missing.append("start_date(如 2025-10-01)")
    return {"parsed": data, "missing_fields": missing}


def route_after_parse(state: TravelState) -> str:
    if state.get("missing_fields") and "destination" in state["missing_fields"]:
        return "clarify"
    return "retrieve"


def clarify_node(state: TravelState) -> Dict[str, Any]:
    parsed = state.get("parsed", {})
    fields = "、".join(state.get("missing_fields", [])) or "目的地"
    return {"reply": f"我可以帮你规划旅行!目前还缺少: {fields}。请补充目的地、出发日期、天数、人均预算与偏好(如美食/历史/海岛),我就会为你生成详细行程。"}


# ---------------------------------------------------------------
# 节点 2:RAG 检索
# ---------------------------------------------------------------
def _build_retriever() -> TravelRetriever:
    retriever = build_retriever()
    if len(retriever) == 0:
        from app.rag.chunker import chunk_documents
        from app.rag.loader import load_kb_documents
        retriever.store.add_documents(chunk_documents(load_kb_documents()))
    return retriever


def retrieve_node(state: TravelState) -> Dict[str, Any]:
    parsed = state.get("parsed", {})
    destination = parsed.get("destination", "")
    prefs = "、".join(parsed.get("preferences", [])) or "经典景点"
    days = parsed.get("days", 3)
    budget = parsed.get("budget_total", 0)
    query = f"目的地 {destination} {days} 天行程,偏好 {prefs}"

    retriever = _build_retriever()
    docs = retriever.search(query=query, city=destination)
    if not docs and destination:
        docs = retriever.search_by_city(destination, k=6)
    # 补全同城景点(行程生成需要尽可能完整的景点清单)
    if destination:
        extra = retriever.search_city_section(destination, "attraction", k=8)
        seen = {d.page_content[:60] for d in docs}
        docs = docs + [d for d in extra if d.page_content[:60] not in seen]
        docs = docs[:10]
    context = TravelRetriever.format_context(docs)
    logger.info("RAG 检索命中 %d 条(城市=%s)", len(docs), destination)
    return {
        "retrieved_docs": [
            {"content": d.page_content[:500], "city": d.metadata.get("city", ""),
             "doc_type": d.metadata.get("doc_type", ""), "score": d.metadata.get("score", 0)}
            for d in docs
        ],
        "context_text": context,
    }


# ---------------------------------------------------------------
# 节点 3:可行性评估
# ---------------------------------------------------------------
def _call_weather_tool(parsed: Dict[str, Any]) -> str:
    destination = parsed.get("destination", "")
    if not destination:
        return "(未识别城市,跳过天气查询)"
    try:
        return get_weather_forecast.invoke({
            "city": destination,
            "start_date": parsed.get("start_date", ""),
            "days": min(max(parsed.get("days", 3), 1), 7),
        })
    except Exception as e:
        return f"(天气工具异常: {e})"


def _budget_fact_sheet(parsed: Dict[str, Any], docs: List) -> str:
    """用预算工具做一次确定性测算,作为评估的事实依据"""
    budget = estimate_budget(
        days=parsed.get("days", 3),
        people=parsed.get("people", 1),
        hotel_per_night=200 if parsed.get("budget_total", 0) >= 1000 else 150,
        meals_per_day=80,
        transport_per_day=30,
        tickets_total=150,
        misc_per_day=30,
    )
    return format_budget(budget)


def evaluate_node(state: TravelState) -> Dict[str, Any]:
    parsed = state.get("parsed", {})
    docs = state.get("retrieved_docs", [])
    weather = _call_weather_tool(parsed)
    budget_fact = _budget_fact_sheet(parsed, docs)

    user_content = (
        (f"【对话历史】\n{format_history(state.get('history'))}\n\n" if state.get("history") else "")
        + f"【用户需求】{state['user_request']}\n"
        f"【知识库检索结果】\n{state.get('context_text', '')[:3000]}\n"
        f"【天气工具数据】\n{weather}\n"
        f"【预算测算】\n{budget_fact}"
    )
    eval_obj = llm_json(EVALUATE_SYSTEM, user_content, EvalReport)
    if eval_obj is None:
        eval_obj = EvalReport(
            weather_ok=True,
            weather_summary=weather,
            season_fit="基于知识库可执行",
            budget_fit=budget_fact,
            crowd_note="节假日人流较大,建议提前预约热门景点",
            risks=["热门景点需提前线上预约"],
            suggestions=["行程安排保持弹性,预留机动时间"],
        )
    return {"evaluation": eval_obj.model_dump(), "weather_data": weather, "tool_data": weather}


class EvalReport(BaseModel):
    weather_ok: bool = Field(default=True, description="天气是否适合出行")
    weather_summary: str = Field(default="", description="天气结论")
    season_fit: str = Field(default="", description="季节匹配度")
    budget_fit: str = Field(default="", description="预算匹配度")
    crowd_note: str = Field(default="", description="人流提示")
    risks: List[str] = Field(default_factory=list, description="风险列表")
    suggestions: List[str] = Field(default_factory=list, description="改进建议")


# ---------------------------------------------------------------
# 节点 4:旅行建议生成
# ---------------------------------------------------------------
def generate_node(state: TravelState) -> Dict[str, Any]:
    parsed = state.get("parsed", {})
    llm = get_llm()

    followup = bool(state.get("prev_plan"))
    if llm_is_mock():
        plan = build_template_plan(parsed, _docs_from_state(state))
        if followup:
            plan += ("\n\n> 💡 演示模式无法对上一版做智能修订,以上为重新生成的模板行程;"
                     "配置 DEEPSEEK_API_KEY 后可基于上一版行程进行个性化调整。")
        return {"plan": plan, "tool_data": state.get("tool_data", "")}

    history_text = format_history(state.get("history"))
    user_prompt = PLAN_USER_TEMPLATE.format(
        req=state["user_request"],
        history=history_text,
        context=state.get("context_text", ""),
        evaluation=str(state.get("evaluation", {})),
        tool_data=state.get("tool_data", "无"),
    ) + "\n\n(可用工具: amap_geocode 地址转坐标、amap_place_search 查景点/餐厅/酒店 POI、get_weather_forecast 查天气。如知识库已覆盖,可不调用工具直接输出)"

    if followup:
        user_prompt = (
            f"【对话历史(较早轮次,保持上下文一致)】\n{history_text}\n\n"
            f"【上一版行程(用户要求在其基础上调整)】\n{state['prev_plan'][:3000]}\n\n"
            f"【修改意见】\n{state['user_request']}\n\n"
            f"请在上一版基础上进行修订:说明你做了哪些调整,并输出完整的新版行程(不要只给差异)。"
        )

    messages = [SystemMessage(content=PLAN_SYSTEM), HumanMessage(content=user_prompt)]
    try:
        final_messages = run_tool_calls(llm, get_amap_tools() + [get_weather_forecast], messages)
        plan = final_messages[-1].content if isinstance(final_messages[-1].content, str) else str(final_messages[-1].content)
        if "MOCK-EMPTY-PLAN" in plan or not plan.strip():
            raise ValueError("LLM 未产出有效行程")
        return {"plan": plan}
    except Exception as e:
        logger.warning("生成节点 LLM 调用失败,回落模板行程: %s", e)
        return {"plan": build_template_plan(parsed, _docs_from_state(state))}


def _docs_from_state(state: TravelState) -> List:
    """将检索到的文档摘要还原为 Document(供模板行程用)"""
    from langchain_core.documents import Document
    docs = []
    for item in state.get("retrieved_docs", []):
        docs.append(
            Document(page_content=item.get("content", ""),
                     metadata={"city": item.get("city", ""), "doc_type": item.get("doc_type", ""),
                               "score": item.get("score", 0), "attraction": _extract_attraction(item.get("content", ""))})
        )
    return docs


def _extract_attraction(content: str) -> str:
    m = re.match(r"^【景点】(.+?)(?:\(|$)", content)
    return m.group(1) if m else ""
