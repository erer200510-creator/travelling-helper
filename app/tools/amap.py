"""
高德地图工具集(Web 服务 API)

对应简历项目经历:
  '通过 MCP 协议调用高德地图进行旅行地点选取,提高 AI 对旅行问题的回答准确度'

说明:
  - 这里以 LangChain Tool 形式封装高德地图 Web 服务 REST 接口,可被 Agent 的
    LLM 按需调用(类似 MCP 的 function calling 语义)。
  - 未配置 AMAP_KEY 时工具会返回明确的错误说明,Agent 会平滑降级到知识库内容。
"""
from __future__ import annotations

import logging
from typing import Dict, List

import requests
from langchain_core.tools import tool

import config

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "ai-travel-agent/1.0"}


def _amap_get(path: str, params: Dict[str, str]) -> Dict:
    """高德 API 请求封装:自动带上 key 与基础校验"""
    if not config.AMAP_KEY:
        return {"status": "0", "info": "AMAP_KEY 未配置:请在 .env 中填写高德开放平台 Web 服务 Key"}
    params = {**params, "key": config.AMAP_KEY}
    try:
        resp = requests.get(f"{config.AMAP_BASE_URL}{path}", params=params, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return {"status": "0", "info": data.get("info", "高德 API 调用失败")}
        return data
    except Exception as e:
        return {"status": "0", "info": f"高德 API 请求异常: {e}"}


@tool
def amap_geocode(address: str, city: str = "") -> str:
    """地理编码:将中文地址(如 '南京')转换为经纬度坐标。
    Args:
        address: 地址或城市名,如 '南京'、'夫子庙'
        city: 可选,限定城市,减少歧义
    Returns:
        包含格式化地址、经度、纬度、城市、行政区划的 JSON 字符串
    """
    data = _amap_get("/v3/geocode/geo", {"address": address, "city": city})
    if data.get("status") != "1":
        return f"[高德工具] {data.get('info')}"
    geocodes = data.get("geocodes", [])
    if not geocodes:
        return f"[高德工具] 未找到地址: {address}"
    g = geocodes[0]
    return (
        f"地址: {g.get('formatted_address')} | 经纬度: {g.get('location')} | "
        f"城市: {g.get('city', '')} | 区: {g.get('district', '')} | 级别: {g.get('level', '')}"
    )


@tool
def amap_place_search(keywords: str, city: str = "") -> str:
    """POI 关键词搜索:查询某城市的景点/餐厅/酒店等地标。
    Args:
        keywords: 搜索关键词,如 '景点'、'网红餐厅'、'希尔顿'
        city: 城市名,如 '西安'
    Returns:
        返回前 5 个命中 POI 的名称、类型、地址与评分
    """
    data = _amap_get("/v3/place/text", {"keywords": keywords, "city": city, "citylimit": "true", "offset": "5", "page": "1", "extensions": "all"})
    if data.get("status") != "1":
        return f"[高德工具] {data.get('info')}"
    pois = data.get("pois", [])
    if not pois:
        return f"[高德工具] 在{city}未找到与'{keywords}'相关的 POI"
    lines = []
    for p in pois[:5]:
        name = p.get("name", "")
        type_ = p.get("type", "")
        addr = p.get("address", "") or p.get("pname", "") + p.get("adname", "")
        dist = p.get("distance", "")
        biz = p.get("biz_ext", {}) or {}
        rating = biz.get("rating", "")
        lines.append(f"- {name} | 类型: {type_} | 地址: {addr} | 距市中心: {dist}米 | 评分: {rating}")
    return "POI 搜索结果:\n" + "\n".join(lines)


@tool
def amap_around_search(location: str, keywords: str = "", radius: int = 3000) -> str:
    """周边查询:以某坐标(经纬度,逗号分隔)为中心搜索周边 POI,用于'景点周边吃什么/住哪'。
    Args:
        location: 中心点经纬度,格式 '经度,纬度',如 '116.397,39.908'
        keywords: 周边关键词,如 '餐厅'、'酒店'、'停车场'
        radius: 搜索半径(米),默认 3000
    Returns:
        返回前 5 个周边 POI 的名称、地址、距离与类型
    """
    data = _amap_get("/v3/place/around", {"location": location, "keywords": keywords, "radius": str(radius), "offset": "5", "page": "1", "extensions": "all"})
    if data.get("status") != "1":
        return f"[高德工具] {data.get('info')}"
    pois = data.get("pois", [])
    if not pois:
        return f"[高德工具] 该位置周边未找到'{keywords}'相关 POI"
    lines = []
    for p in pois[:5]:
        lines.append(
            f"- {p.get('name', '')} | 类型: {p.get('type', '')} | "
            f"距离: {p.get('distance', '')}米 | 地址: {p.get('address', '')}"
        )
    return f"以 {location} 为中心 {radius}m 内搜索结果:\n" + "\n".join(lines)


@tool
def amap_regeo(location: str) -> str:
    """逆地理编码:经纬度 → 结构化地址(判断该地属于哪个景区/商圈)。
    Args:
        location: 经纬度,格式 '经度,纬度'
    Returns:
        格式化地址、城市、商圈/街道信息
    """
    data = _amap_get("/v3/geocode/regeo", {"location": location, "extensions": "base"})
    if data.get("status") != "1":
        return f"[高德工具] {data.get('info')}"
    rc = data.get("regeocode", {})
    addr = rc.get("formatted_address", "")
    comp = rc.get("addressComponent", {})
    return f"地址: {addr} | 城市: {comp.get('city', '')} | 区: {comp.get('district', '')} | 商圈: {comp.get('businessAreas', '')}"


def get_amap_tools() -> List:
    """返回全部高德工具(供 LLM bind_tools)"""
    return [amap_geocode, amap_place_search, amap_around_search, amap_regeo]


def amap_available() -> bool:
    return bool(config.AMAP_KEY)
