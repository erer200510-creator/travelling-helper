"""
AI 旅行 Agent - Streamlit 网页对话框

运行:
  streamlit run app/ui/streamlit_app.py

功能:
  - 聊天气泡式对话(用户/助手),Markdown 渲染行程
  - 多轮对话:基于上一版行程直接追问修改(如"预算降到2000""第2天轻松点")
  - 每次回答展示处理过程:需求解析 → RAG 检索 → 可行性评估 → 行程生成
  - 侧边栏:快捷表单、知识库状态、新建对话
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 屏蔽 langgraph 内部与业务无关的废弃告警
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except Exception:
    pass

import streamlit as st

from app.agent.graph import stream_agent
from app.llm import llm_is_mock
from app.rag.retriever import build_retriever

st.set_page_config(page_title="AI 旅行 Agent", page_icon="🧭", layout="wide")

# ---------------------------------------------------------------
# 会话状态
# ---------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content, trace?, docs?, is_plan?}]

# ---------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------
with st.sidebar:
    st.title("🧭 AI 旅行 Agent")

    if llm_is_mock():
        st.warning("⚠️ 演示模式\n未配置 DEEPSEEK_API_KEY,行程由知识库模板生成。配置后支持语义解析与个性化修订。")
    else:
        st.success("✅ 已接入 DeepSeek 大模型")

    # 知识库状态
    try:
        retriever = build_retriever()
        st.info(f"📚 知识库: {len(retriever)} 条旅行案例片段")
    except Exception:
        st.info("📚 知识库: 未构建(运行 scripts/build_kb.py)")

    if st.button("🆕 新建对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("⚡ 快速填写需求")
    with st.form("quick_form", clear_on_submit=True):
        dest = st.text_input("目的地", placeholder="如: 西安")
        days = st.number_input("天数", min_value=1, max_value=14, value=4)
        budget = st.number_input("人均预算(元)", min_value=0, max_value=50000, value=3000, step=100)
        prefs = st.multiselect("偏好", ["历史", "美食", "自然风光", "海岛", "都市夜景", "亲子", "省钱", "博物馆"])
        submitted = st.form_submit_button("🧳 一键生成", use_container_width=True)
        if submitted and dest:
            req = f"计划去{dest}玩{days}天,人均预算{budget}元"
            if prefs:
                req += ",喜欢" + "、".join(prefs)
            # 仅置待处理请求;本轮消息由主流程统一渲染,避免 history 重复
            st.session_state._pending = req
            st.rerun()

    st.divider()
    st.caption("多轮对话提示:回复结果后可直接说\n'预算降到 2000' / '第 2 天放松一点' 来修订行程")

# ---------------------------------------------------------------
# 主区域:对话展示
# ---------------------------------------------------------------
st.title("🧭 AI 旅行助手")
st.caption("一句话,帮你规划完整行程:目的地 + 日期 + 天数 + 预算 + 偏好")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🧭"):
        st.markdown(msg["content"])
        if msg.get("trace"):
            with st.expander("🔍 处理过程", expanded=False):
                for line in msg["trace"]:
                    st.markdown(line)
                if msg.get("docs"):
                    st.caption(f"📚 本次命中 {len(msg['docs'])} 条知识库片段:")
                    st.code("\n".join(msg["docs"][:8]))

# ---------------------------------------------------------------
# 输入与执行
# ---------------------------------------------------------------
user_input = st.chat_input("描述你的旅行需求,如: 国庆去西安玩4天,人均3000,喜欢历史和美食")
pending = user_input or st.session_state.pop("_pending", "")

if pending:
    # 多轮:上一条助手消息是行程 → 本次为修订模式
    prev_plan = ""
    is_followup = False
    if st.session_state.messages:
        last = st.session_state.messages[-1]
        if last.get("role") == "assistant" and last.get("is_plan"):
            prev_plan = last["content"]
            is_followup = True

    # 对话历史(不含本轮):跨轮上下文记忆,让 Agent 记得目的地/偏好/预算
    history = [
        {"role": m["role"], "content": m["content"], "is_plan": bool(m.get("is_plan"))}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": pending})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(pending)

    with st.chat_message("assistant", avatar="🧭"):
        status = st.status("正在规划行程...", expanded=True)
        if is_followup:
            status.write("🔄 检测到多轮对话:将在上一版行程基础上修订")
        elif history:
            status.write(f"🧠 已带上 {len(history)} 条对话历史(上下文记忆)")

        trace, docs = [], []
        plan, reply = "", ""
        for update in stream_agent(pending, prev_plan=prev_plan, is_followup=is_followup, history=history):
            for node, data in update.items():
                if node == "parse" and data.get("parsed"):
                    p = data["parsed"]
                    line = f"① 需求解析 ✔ 目的地 **{p.get('destination')}** | {p.get('days')} 天 | 人均 {p.get('budget_total')} 元 | 偏好 {p.get('preferences')}"
                    status.write(line)
                    trace.append(f"① 需求解析: 目的地 {p.get('destination')}, {p.get('days')}天, 预算 {p.get('budget_total')}元, 偏好 {p.get('preferences')}")
                elif node == "clarify" and data.get("reply"):
                    reply = data["reply"]
                    status.write("❓ 信息不足,需要进一步确认")
                elif node == "retrieve":
                    n = len(data.get("retrieved_docs", []))
                    status.write(f"② RAG 检索 ✔ 命中 {n} 条知识库片段")
                    trace.append(f"② RAG 检索: 命中 {n} 条(景点/美食/住宿/预算/可行性)")
                    docs = [
                        f"[{d.get('city')} · {d.get('doc_type')} · {d.get('score'):.3f}] {d.get('content','')[:60]!r}"
                        for d in data.get("retrieved_docs", [])
                    ]
                elif node == "evaluate":
                    ev = data.get("evaluation", {})
                    status.write(f"③ 可行性评估 ✔ 天气OK:{ev.get('weather_ok')} | {str(ev.get('weather_summary',''))[:40]}")
                    trace.append(f"③ 可行性评估: 天气OK={ev.get('weather_ok')} | {ev.get('weather_summary','')} | 风险{len(ev.get('risks',[]))}条")
                elif node == "generate":
                    plan = data.get("plan", "")
                    status.write("④ 行程生成 ✔")
                    trace.append("④ 行程生成: 完成")

        status.update(label="规划完成 ✅" if plan or reply else "未生成结果", state="complete", expanded=False)

        if plan:
            st.markdown(plan)
            st.session_state.messages.append({"role": "assistant", "content": plan, "trace": trace, "docs": docs, "is_plan": True})
        else:
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply, "trace": trace, "is_plan": False})
