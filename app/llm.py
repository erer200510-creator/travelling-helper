"""
LLM 接入层

对应简历项目经历:
  '接入 DeepSeek API 实现智能问答能力 / Prompt 优化'

说明:
  1. 配置 DEEPSEEK_API_KEY 后,通过 langchain-deepseek 官方接入 deepseek-chat;
  2. 未配置 Key 时自动降级为 MockTravelLLM(规则演示),保证整个 Agent 流程
     离线可跑通、可演示、可测试;
  3. 提供 llm_json() 统一结构化输出解析:LLM 输出 JSON → 校验 → 修复重试。
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# 工厂:获取真实/模拟 LLM
# ---------------------------------------------------------------
@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    provider = config.LLM_PROVIDER.lower()
    if provider == "deepseek" or (provider == "auto" and config.DEEPSEEK_API_KEY):
        from langchain_deepseek import ChatDeepSeek
        llm = ChatDeepSeek(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        logger.info("LLM provider: DeepSeek (%s)", config.DEEPSEEK_MODEL)
        return llm
    logger.info("LLM provider: MOCK(演示模式,未配置 DEEPSEEK_API_KEY)")
    return MockTravelLLM()


def llm_is_mock() -> bool:
    return isinstance(get_llm(), MockTravelLLM)


# ---------------------------------------------------------------
# 结构化输出工具:LLM → JSON → Pydantic 校验(带一次修复重试)
# ---------------------------------------------------------------
def extract_json(text: str) -> Optional[Dict]:
    """从 LLM 文本中提取第一个合法 JSON 对象"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 提取首个 { ... } 块(处理前后缀文字)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def llm_json(system: str, user: str, schema: Type[BaseModel],
             attempts: int = 2) -> Optional[BaseModel]:
    """调用 LLM 并解析为 Pydantic 对象;失败时带纠错提示重试一次"""
    llm = get_llm()
    prompt = (f"【系统】\n{system}\n\n【用户】\n{user}")
    for i in range(attempts):
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])  # type: ignore
            data = extract_json(resp.content if isinstance(resp.content, str) else str(resp.content))
            if data is not None and isinstance(data, dict):
                # 容错:LLM 返回 null 的字段按 Schema 默认值处理
                data = {k: v for k, v in data.items() if v is not None}
                return schema.model_validate(data)
        except Exception as e:
            logger.warning("llm_json 第 %d 次失败: %s", i + 1, e)
        if i == 0:
            user = user + "\n\n(提示:上一次输出无法解析。请只输出一个合法的 JSON 对象,不要输出任何其他文字。)"
    return None


# ---------------------------------------------------------------
# MockTravelLLM:离线演示用规则模型
# ---------------------------------------------------------------
def _guess_city(text: str) -> str:
    cities = ["北京", "上海", "杭州", "南京", "西安", "成都", "重庆", "厦门", "广州", "长沙", "青岛", "武汉", "天津", "苏州", "大理", "丽江"]
    for c in cities:
        if c in text:
            return c
    m = re.search(r"(?:去|到|在|玩|游|目的地|城市)(?:.*?)([\u4e00-\u9fff]{2,4})", text)
    if m:
        return m.group(1)
    return "西安"


def _guess_days(text: str) -> int:
    m = re.search(r"(\d+)\s*天", text)
    return int(m.group(1)) if m else 4


def _guess_budget(text: str) -> int:
    m = re.search(r"(?:人均\s*)?(\d{3,5})\s*元", text) or re.search(r"人均\s*(\d{3,5})", text)
    return int(m.group(1)) if m else 0


class MockTravelLLM(BaseChatModel):
    """规则驱动的演示 LLM:让无 Key 环境也能完整走通 Agent 流程"""

    @property
    def _llm_type(self) -> str:
        return "mock-travel-llm"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> ChatResult:
        user = "\n".join(m.content for m in messages if isinstance(m, (HumanMessage, SystemMessage)) if isinstance(m.content, str))
        if "需求分析助手" in user:
            content = self._parse_response(user)
        elif "可行性评估专家" in user:
            content = self._evaluate_response(user)
        else:
            content = self._plan_hint(user)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    # ---- 需求解析节点 ----
    def _parse_response(self, user_text: str) -> str:
        # 只解析用户真实输入片段(【用户】之后),避免模板枚举词干扰
        req = user_text.split("【用户】")[-1]
        city = _guess_city(req)
        days = _guess_days(req)
        budget = _guess_budget(req)
        prefs: List[str] = []
        for kw, tag in [("美食", "美食"), ("火锅", "美食"), ("小吃", "美食"), ("自然", "自然风光"), ("爬山", "自然风光"),
                        ("历史", "历史文化"), ("古迹", "历史文化"), ("海边", "海岛"), ("海岛", "海岛"),
                        ("情侣", "情侣/浪漫"), ("省钱", "经济"), ("穷游", "经济"), ("亲子", "亲子"), ("熊猫", "亲子")]:
            if kw in req:
                prefs.append(tag)
        if not prefs:
            prefs.append("经典必去")
        date_match = re.search(r"(\d{4}年)?(\d{1,2})月(\d{1,2})日", req)
        start_date = f"2025-{int(date_match.group(2)):02d}-{int(date_match.group(3)):02d}" if date_match else ""
        return json.dumps({
            "destination": city, "start_date": start_date, "days": days,
            "budget_total": budget, "people": 1,
            "preferences": prefs, "transport_mode": "",
            "notes": f"(mock 解析)mock 演示,字段由规则正则提取,配置 DeepSeek Key 后为语义解析",
        }, ensure_ascii=False)

    # ---- 可行性评估节点 ----
    def _evaluate_response(self, user_text: str) -> str:
        city = _guess_city(user_text.split("【用户需求】")[-1])
        return json.dumps({
            "weather_ok": True, "weather_summary": f"{city} 近期气温适中(mock 模式未调用天气接口)",
            "season_fit": "季节匹配度良好,与该案例建议出行时间一致",
            "budget_fit": "预算与知识库样本基本匹配",
            "crowd_note": "节假日人流较大,建议错峰并提前预约热门景点",
            "risks": ["热门景点(博物馆/博物院类)需提前预约", "山区/海边行程需关注天气"],
            "suggestions": ["第一天下午以市区轻度行程为主,保留体力", "把最远的景点放在行程中段"],
        }, ensure_ascii=False)

    # ---- 行程生成节点(mock 给出结构化提示,真正文字由 fallback_plan 生成)----
    def _plan_hint(self, user_text: str) -> str:
        return (
            "【MOCK-EMPTY-PLAN】当前为演示模式:请使用 fallback_plan.build_template_plan 生成示例行程。"
            "配置 DEEPSEEK_API_KEY 后将由 DeepSeek 生成完整行程。"
        )


# ---------------------------------------------------------------
# 工具调用辅助:在 Agent 节点内执行 LLM 发起的 tool_calls
# ---------------------------------------------------------------
def run_tool_calls(llm: BaseChatModel, tools: List, messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    让 LLM 绑定工具并最多做 2 轮 tool 调用循环,返回最终 LLM 消息。
    用于行程生成节点的工具增强(高德/天气)。
    """
    if llm_is_mock() or not tools:
        return messages
    tool_map = {t.name: t for t in tools}
    l = llm.bind_tools(tools)  # type: ignore
    msgs = list(messages)
    for _ in range(2):
        ai = l.invoke(msgs)  # type: ignore
        msgs.append(ai)
        tool_calls = getattr(ai, "tool_calls", None)
        if not tool_calls:
            break
        for tc in tool_calls:
            t = tool_map.get(tc["name"])
            if t is None:
                continue
            try:
                result = t.invoke(tc["args"])
            except Exception as e:
                result = f"[工具执行失败] {e}"
            from langchain_core.messages import ToolMessage
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return msgs
