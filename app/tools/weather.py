"""
天气工具(Open-Meteo,免费无 Key)

用于可行性评估节点:结合旅行日期给出目的地的温度/降水预报,
让 Agent 判断'这个时间去合不合适'。

对应简历项目经历中的 '可行性评估' 环节。
"""
from __future__ import annotations

from typing import Dict

import requests
from langchain_core.tools import tool

# WMO weather code → 中文描述(全球通用编码)
_WMO_ZH = {
    0: "晴", 1: "基本晴", 2: "晴间多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "小雨", 55: "中雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    80: "阵雨", 81: "中阵雨", 82: "强阵雨",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
}

_HEADERS = {"User-Agent": "ai-travel-agent/1.0"}


def _geocode(city: str) -> Dict:
    """Open-Meteo 城市地理编码(支持中文城市名)"""
    resp = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 3, "language": "zh"},
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return {}
    best = results[0]
    return {"name": best.get("name", city), "latitude": best["latitude"], "longitude": best["longitude"],
            "country": best.get("country", "")}


@tool
def get_weather_forecast(city: str, start_date: str = "", days: int = 4) -> str:
    """查询某城市未来几天的天气预报(免费 Open-Meteo,无需 Key)。
    Args:
        city: 城市中文名,如 '西安'
        start_date: 开始日期,格式 YYYY-MM-DD;留空则从今天开始
        days: 预报天数(1-7),默认 4
    Returns:
        逐日天气:日期、最高/最低温、降水概率、天气现象
    """
    try:
        geo = _geocode(city)
        if not geo:
            return f"[天气工具] 未识别城市:{city}(请检查城市名)"
        params = {
            "latitude": geo["latitude"], "longitude": geo["longitude"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "timezone": "Asia/Shanghai", "forecast_days": max(1, min(int(days or 4), 7)),
        }
        if start_date:
            from datetime import date, timedelta
            try:
                d = date.fromisoformat(start_date)
                params.update({
                    "start_date": d.isoformat(),
                    "end_date": (d + timedelta(days=max(1, min(int(days or 4), 7)) - 1)).isoformat(),
                })
                params.pop("forecast_days", None)
            except ValueError:
                pass
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        dates = daily.get("time", [])
        if not dates:
            return f"[天气工具] {city} 暂无预报数据"
        lines = []
        for i, d in enumerate(dates):
            code = daily["weather_code"][i]
            desc = _WMO_ZH.get(code, f"天气码{code}")
            lines.append(
                f"- {d}: {desc} | 最高 {daily['temperature_2m_max'][i]}°C | "
                f"最低 {daily['temperature_2m_min'][i]}°C | 降水概率 {daily['precipitation_probability_max'][i]}%"
            )
        return f"{geo['name']}({geo.get('country','')}) 天气预报:\n" + "\n".join(lines)
    except Exception as e:
        return f"[天气工具] 请求异常: {e}"


def get_weather_tools():
    return [get_weather_forecast]
