"""
兜底行程生成器(模板式)

用途:
  1. 演示模式(Mock LLM)下生成可读的示例行程;
  2. 真实 LLM 调用失败时的降级方案,保证功能闭环。

数据来源:完全来自 RAG 检索结果,不做臆造。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


def _body(content: str, default: str = "") -> str:
    """取文档第一行标题之后的内容作为正文摘录"""
    parts = content.split("\n", 1)
    return parts[1].strip().replace("\n", "; ")[:300] if len(parts) > 1 and parts[1].strip() else default


def build_template_plan(parsed: Dict[str, Any], docs: List) -> str:
    """从检索结果生成模板行程(演示/兜底)"""
    city = parsed.get("destination") or "目的地"
    days = max(1, int(parsed.get("days") or 3))
    people = max(1, int(parsed.get("people") or 1))
    budget_total = int(parsed.get("budget_total") or 0)

    attracted: List[str] = []
    foods: List[str] = []
    hotels: List[str] = []
    budget_note = ""
    tips: List[str] = []

    for d in docs:
        meta = d.metadata or {}
        content = d.page_content
        if meta.get("doc_type") == "attraction":
            name = meta.get("attraction") or re.sub(r"^【景点】(.*?)(\(|$)", r"\1", content.split("\n", 1)[0])
            attracted.append(name)
        elif meta.get("doc_type") == "food":
            foods.append(_body(content))
        elif meta.get("doc_type") == "hotel":
            hotels.append(_body(content))
        elif meta.get("doc_type") == "budget":
            budget_note = _body(content, content)
        elif meta.get("doc_type") == "feasibility":
            for line in content.split("\n"):
                if line.startswith("风险") or line.startswith("建议") or line.startswith("最佳"):
                    tips.append(line[line.find(":") + 1:].strip()) if ":" in line else tips.append(line)

    # 去重(按出现顺序)
    def dedup(seq):
        seen, out = set(), []
        for s in seq:
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    attracted = dedup(attracted)[:days * 2]
    if not attracted:
        attracted = [f"{city} 市区经典景点", f"{city} 博物馆/老街", f"{city} 周边自然景点"]

    # 按天分配景点
    per_day = max(1, -(-len(attracted) // days))
    days_plan = []
    for i in range(days):
        chunk = attracted[i * per_day:(i + 1) * per_day]
        if not chunk:
            chunk = [f"{city} 自由活动 / 回程准备"]
        plan_lines = [
            f"### Day{i + 1}",
            f"- 上午: {chunk[0]}",
        ]
        if len(chunk) > 1:
            plan_lines.append(f"- 下午: {chunk[1]}")
        else:
            plan_lines.append("- 下午: 就近商圈自由活动 / 小吃觅食")
        plan_lines.append(f"- 晚上: {city} 夜市/景区夜景,或看一场演出")
        days_plan.append("\n".join(plan_lines))

    food_txt = foods[0] if foods else f"当地招牌菜、小吃街(详见知识库 {city} 美食指南)"
    hotel_txt = hotels[0] if hotels else f"{city} 市区中心区经济型酒店/青旅,人均 100-200 元/晚"

    total_hint = f",总预算约 {budget_total} 元" if budget_total else ""

    tips_txt = "\n".join(f"- {t}" for t in tips[:4]) if tips else f"- 热门景点建议线上提前购票/预约;- {city} 春秋天气最宜人,夏季注意防晒"

    return f"""# {city} {days} 日游行程(示例计划 · 由知识库检索生成)

> 说明:本行程由模板引擎基于知识库检索结果拼装(演示/兜底模式)。
> 配置 DEEPSEEK_API_KEY 后将由大模型生成更个性化的完整行程。

## 行程安排
{chr(10).join(days_plan)}

## 餐饮推荐
- {food_txt}

## 住宿建议
- {hotel_txt}

## 预算参考
- {budget_note or f"{city} 经济型日均约 200-300 元/人{total_hint}"}
- 人均总预算约 {(budget_total if budget_total else 4 * 250)} 元(根据 RAG 知识库日均参考估算)

## 出行贴士
{tips_txt}
"""
