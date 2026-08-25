"""
命令行交互入口

用法:
  python -m app.ui.cli                          # 交互式对话
  python -m app.ui.cli --demo "国庆想去西安玩4天,人均3000,喜欢历史美食"
  python -m app.ui.cli --build-kb               # 仅重建知识库
"""
from __future__ import annotations

import argparse
import sys

# 保证以脚本方式运行时能 import app 包
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging
import warnings

# 屏蔽 langgraph 内部与业务无关的废弃告警
try:
    from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)
except Exception:
    pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

from app.agent.graph import stream_agent  # noqa: E402
from app.llm import llm_is_mock  # noqa: E402


def _print_step(step_name: str, update: dict) -> None:
    """打印单个节点的执行摘要"""
    if step_name == "parse" and "parsed" in update:
        p = update["parsed"]
        print("  [1/4] 需求解析:", end=" ")
        if p:
            print(f"目的地 {p.get('destination')} | {p.get('days')} 天 | 人均预算 {p.get('budget_total')} 元 | 偏好 {p.get('preferences')}")
        else:
            print("解析失败(将询问补充信息)")
    elif step_name == "clarify" and "reply" in update:
        print(f"  [!] 询问补充: {update['reply']}")
    elif step_name == "retrieve" and "context_text" in update:
        n = len(update.get("retrieved_docs", []))
        print(f"  [2/4] RAG 检索: 命中 {n} 条知识库片段")
    elif step_name == "evaluate" and "evaluation" in update:
        print(f"  [3/4] 可行性评估: 天气OK={update['evaluation'].get('weather_ok')} | {update['evaluation'].get('weather_summary','')[:50]}")
    elif step_name == "generate" and "plan" in update:
        print("  [4/4] 行程生成: 完成")


def run_demo(user_request: str, history: list | None = None) -> list:
    """执行一轮对话;history 为跨轮上下文记忆(会被追加最新一轮)"""
    history = history if history is not None else []
    print(f"\n>>> 用户: {user_request}\n")
    if history:
        print(f"  (已携带 {len(history)} 条对话历史)")
    final = None
    for update in stream_agent(user_request, history=history):
        for node, data in update.items():
            _print_step(node, data)
            final = data
    print("\n" + "=" * 60)
    output = ""
    if final and final.get("reply"):
        output = final["reply"]
        print(output)
    elif final and final.get("plan"):
        output = final["plan"]
        print(output)
    else:
        print("(未生成结果,请检查日志)")
    print("=" * 60)
    history.append({"role": "user", "content": user_request})
    history.append({"role": "assistant", "content": output, "is_plan": bool(final and final.get("plan"))})
    return history


def run_interactive() -> None:
    print("AI 旅行 Agent 已启动(输入 exit/quit 退出)")
    if llm_is_mock():
        print("⚠ 当前为演示模式:未配置 DEEPSEEK_API_KEY,行程将由模板引擎生成。")
    print("🧠 已启用对话上下文记忆:后续轮次可直接说'预算降到2000'、'第2天轻松点'")
    history: list = []
    while True:
        try:
            req = input("\n请输入旅行需求> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break
        if req.lower() in ("exit", "quit", "q"):
            print("再见!")
            break
        if not req:
            continue
        history = run_demo(req, history)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI 旅行 Agent")
    parser.add_argument("--demo", type=str, help="单次执行:直接给出一条旅行需求")
    parser.add_argument("--build-kb", action="store_true", help="重建 RAG 知识库")
    args = parser.parse_args()

    if args.build_kb:
        from scripts.build_kb import main as build_main
        build_main()
        return
    if args.demo:
        run_demo(args.demo)
        return
    run_interactive()


if __name__ == "__main__":
    main()
