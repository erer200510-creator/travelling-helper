"""
RAG 数据加载与清洗层

对应简历项目经历:
  '利用 Python 脚本完成对旅行案例的 JSON 格式数据的清洗格式化,构建高质量向量数据库'

职责:
  1. 读取 data/travel_cases.json
  2. 字段归一化 / 去空 / 去重 / 类型矫正
  3. 输出标准化的 Document 列表(统一 langchain Document 格式,便于后续切分与入库)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document

import config


def _clean_text(value: Any) -> str:
    """去除多余空白、全角空格,规整换行,返回单行友好文本"""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _clean_str_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out = []
    for v in values:
        v = _clean_text(v)
        if v and v not in out:
            out.append(v)
    return out


def _clean_attractions(attractions: Any) -> List[Dict[str, str]]:
    """景点清洗:只保留字段名合法且非空的条目"""
    if not isinstance(attractions, list):
        return []
    keys = ("name", "type", "hours", "ticket", "duration", "description", "tips")
    cleaned = []
    for item in attractions:
        if not isinstance(item, dict):
            continue
        doc = {k: _clean_text(item.get(k, "")) for k in keys}
        if not doc["name"]:
            continue
        doc.pop("", None)
        cleaned.append({k: v for k, v in doc.items() if v})
    return cleaned


def _clean_food(food: Any) -> Dict[str, Any]:
    if not isinstance(food, dict):
        return {"must_eat": [], "restaurants": []}
    res = []
    for r in food.get("restaurants", []) or []:
        if not isinstance(r, dict) or not _clean_text(r.get("name", "")):
            continue
        res.append({
            "name": _clean_text(r.get("name")),
            "dish": _clean_text(r.get("dish")),
            "price_range": _clean_text(r.get("price_range")),
            "area": _clean_text(r.get("area")),
        })
    return {
        "must_eat": _clean_str_list(food.get("must_eat")),
        "restaurants": res,
    }


def _clean_hotels(hotels: Any) -> Dict[str, Any]:
    if not isinstance(hotels, dict):
        return {"recommended": [], "strategy": ""}
    items = []
    for h in hotels.get("recommended", []) or []:
        if not isinstance(h, dict) or not _clean_text(h.get("name", "")):
            continue
        items.append({
            "name": _clean_text(h.get("name")),
            "area": _clean_text(h.get("area")),
            "price_range": _clean_text(h.get("price_range")),
            "features": _clean_text(h.get("features")),
        })
    return {
        "recommended": items,
        "strategy": _clean_text(hotels.get("strategy")),
    }


def _clean_case(raw: Dict[str, Any]) -> Dict[str, Any]:
    """单条旅行案例的归一化"""
    cleaned = {
        "id": _clean_text(raw.get("id")),
        "city": _clean_text(raw.get("city")),
        "season": _clean_text(raw.get("season")) or "全年",
        "days": int(raw.get("days") or 0),
        "tags": _clean_str_list(raw.get("tags")),
        "overview": _clean_text(raw.get("overview")),
        "highlights": _clean_str_list(raw.get("highlights")),
        "attractions": _clean_attractions(raw.get("attractions")),
        "food": _clean_food(raw.get("food")),
        "hotels": _clean_hotels(raw.get("hotels")),
        "transport": raw.get("transport") if isinstance(raw.get("transport"), dict) else {},
        "budget": raw.get("budget") if isinstance(raw.get("budget"), dict) else {},
        "feasibility": raw.get("feasibility") if isinstance(raw.get("feasibility"), dict) else {},
    }
    return cleaned


def load_and_clean_cases(path: Path | None = None) -> List[Dict[str, Any]]:
    """
    加载原始 JSON 案例并清洗。

    Returns:
        List[Dict] 标准化后的案例列表(字段结构见 _clean_case)
    """
    path = path or config.KB_FILE
    if not path.exists():
        raise FileNotFoundError(f"知识库文件不存在: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("知识库文件顶层必须是 JSON 数组")

    cases = [_clean_case(c) for c in raw if isinstance(c, dict)]
    # 去重:同一 id / 同 city+days 只保留第一条
    seen_ids, seen_city_days, deduped = set(), set(), []
    for c in cases:
        key = c["id"] or f"{c['city']}-{c['days']}"
        if key in seen_ids:
            continue
        seen_ids.add(key)
        if (c["city"], c["days"]) in seen_city_days:
            continue
        seen_city_days.add((c["city"], c["days"]))
        deduped.append(c)
    return deduped


def case_to_documents(case: Dict[str, Any]) -> List[Document]:
    """
    将一条清洗后的案例转换为多条语义完整的 Document。
    直接返回 Document 列表,由 chunker 决定是否二次切分(长文本才切)。
    """
    city = case["city"]
    season = case["season"]
    days = case["days"]
    docs: List[Document] = []
    meta = {"source": case["id"], "city": city, "season": season, "days": days}

    # 1. 总览文档
    if case["overview"]:
        headline = "、".join(case["highlights"]) or "详见正文"
        content = (
            f"【{city}{season}{days}日游·总览】\n"
            f"城市:{city} | 适合季节:{season} | 建议天数:{days}天 | 主题标签:{'、'.join(case['tags'])}\n"
            f"行程亮点:{headline}\n"
            f"概述:{case['overview']}"
        )
        docs.append(Document(page_content=content, metadata={**meta, "doc_type": "overview"}))

    # 2. 每个景点一份文档(粒度适中,检索命中更准)
    for a in case["attractions"]:
        parts = [f"【景点】{a['name']}({city})", f"类型:{a.get('type','')} | 开放时间:{a.get('hours','')} | 门票:{a.get('ticket','')} | 建议游玩:{a.get('duration','')}"]
        if a.get("description"):
            parts.append(f"介绍:{a['description']}")
        if a.get("tips"):
            parts.append(f"小贴士:{a['tips']}")
        docs.append(Document(page_content="\n".join(parts), metadata={**meta, "doc_type": "attraction", "attraction": a["name"]}))

    # 3. 美食文档
    food = case["food"]
    if food:
        parts = [f"【美食】{city}特色美食指南"]
        if food.get("must_eat"):
            parts.append(f"必吃:{'、'.join(food['must_eat'])}")
        for r in food.get("restaurants", []):
            parts.append(f"- {r['name']}({r.get('area','')}) 招牌:{r.get('dish','')} 人均:{r.get('price_range','')}")
        docs.append(Document(page_content="\n".join(parts), metadata={**meta, "doc_type": "food"}))

    # 4. 住宿文档
    hotels = case["hotels"]
    if hotels:
        parts = [f"【住宿】{city}住宿推荐"]
        for h in hotels.get("recommended", []):
            parts.append(f"- {h['name']}({h.get('area','')}) {h.get('price_range','')} {h.get('features','')}")
        if hotels.get("strategy"):
            parts.append(f"策略:{hotels['strategy']}")
        docs.append(Document(page_content="\n".join(parts), metadata={**meta, "doc_type": "hotel"}))

    # 5. 预算文档
    budget = case["budget"]
    if budget:
        parts = [f"【预算】{city}{days}日游({budget.get('style','')})预算参考"]
        if budget.get("per_day"):
            parts.append(f"日均参考:约 {budget['per_day']} 元/人/天")
        if budget.get("breakdown"):
            parts.append(f"构成: {budget['breakdown']}")
        docs.append(Document(page_content="\n".join(parts), metadata={**meta, "doc_type": "budget"}))

    # 6. 可行性文档(天气/人流/风险 → 直接服务可行性评估节点)
    feas = case["feasibility"]
    if feas:
        parts = [f"【可行性评估】{city} {season} 出行可行性"]
        if feas.get("best_time"):
            parts.append(f"最佳时间:{feas['best_time']}")
        if feas.get("weather"):
            parts.append(f"天气:{feas['weather']}")
        if feas.get("crowd"):
            parts.append(f"人流:{feas['crowd']}")
        if feas.get("risks"):
            parts.append(f"风险提示:{'；'.join(feas['risks'])}")
        if feas.get("suggestions"):
            parts.append(f"建议:{'；'.join(feas['suggestions'])}")
        docs.append(Document(page_content="\n".join(parts), metadata={**meta, "doc_type": "feasibility"}))

    # 7. 交通文档
    transport = case["transport"]
    if transport:
        parts = [f"【交通】{city}交通指南"]
        if transport.get("arrival"):
            parts.append(f"到达:{transport['arrival']}")
        if transport.get("in_city"):
            parts.append(f"市内:{transport['in_city']}")
        docs.append(Document(page_content="\n".join(parts), metadata={**meta, "doc_type": "transport"}))

    return docs


def load_kb_documents(path: Path | None = None) -> List[Document]:
    """一站式:加载 JSON → 清洗 → 转 Document"""
    cases = load_and_clean_cases(path)
    docs: List[Document] = []
    for case in cases:
        docs.extend(case_to_documents(case))
    return docs
