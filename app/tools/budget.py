"""
预算测算工具(纯函数计算,无外部依赖)

对应用户需求:生成包含'每日景点、餐饮、酒店及预算'的详细行程计划。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class BudgetResult:
    style: str             # 经济 / 舒适 / 豪华
    days: int
    people: int
    hotel_total: float     # 住宿总价
    meal_total: float      # 餐饮总价
    transport_total: float  # 市内交通总价
    ticket_total: float    # 门票总价
    misc_total: float      # 机动/购物
    total: float           # 总预算
    per_person: float      # 人均


def estimate_budget(days: int = 3, people: int = 1,
                    hotel_per_night: float = 200, meals_per_day: float = 80,
                    transport_per_day: float = 30, tickets_total: float = 150,
                    misc_per_day: float = 30, style: str = "经济") -> Dict:
    """按天估算总预算。

    Args:
        days: 行程天数
        people: 人数(酒店按间夜,不按人头翻倍)
        hotel_per_night: 每人间夜房价
        meals_per_day: 每人每天餐饮
        transport_per_day: 每人每天市内交通
        tickets_total: 每人全程门票合计
        misc_per_day: 每人每天机动支出
        style: 经济/舒适/豪华(仅用于文案)
    """
    days = max(1, int(days))
    people = max(1, int(people))
    hotel_total = hotel_per_night * days
    meal_total = meals_per_day * days * people
    transport_total = transport_per_day * days * people
    ticket_total = tickets_total * people
    misc_total = misc_per_day * days * people
    total = hotel_total + meal_total + transport_total + ticket_total + misc_total
    return asdict(BudgetResult(
        style=style, days=days, people=people,
        hotel_total=round(hotel_total, 1), meal_total=round(meal_total, 1),
        transport_total=round(transport_total, 1), ticket_total=round(ticket_total, 1),
        misc_total=round(misc_total, 1), total=round(total, 1),
        per_person=round(total / people, 1),
    ))


def format_budget(budget: Dict) -> str:
    """预算结果 → 可读文本(供 prompt 引用)"""
    return (
        f"预算测算({budget['style']}型,{budget['days']}天{budget['people']}人):\n"
        f"住宿 {budget['hotel_total']} + 餐饮 {budget['meal_total']} + 市内交通 {budget['transport_total']}"
        f" + 门票 {budget['ticket_total']} + 机动 {budget['misc_total']}\n"
        f"总预算: {budget['total']} 元 | 人均: {budget['per_person']} 元"
    )
