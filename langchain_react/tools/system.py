"""系统类工具 —— P1 用来验证"工具调用链路是否真的通"。

current_time   ：离线也能验证（无需 7888、无需任何后端服务）
service_health ：探测各微服务是否在线，Agent 排障时第一个该调的工具
"""

from __future__ import annotations

import json
from datetime import datetime

from langchain_core.tools import tool

from qc_agent_mcp.services import SERVICE_TABLE, health_url
from qc_agent_mcp.tools.http_client import get_json


@tool
def current_time(_: str = "") -> str:
    """获取当前日期与时间。当你需要知道"今天是什么时候"时使用。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def service_health(name: str = "report_generation") -> str:
    """查询 QC 各微服务是否在线。

    name 取值：llm_proxy / tool_recommend / report_generation / qc_review /
    general_qa / ai_review / intent_judgment / report_optimization / file_middleware
    """
    try:
        url = health_url(name)
    except KeyError as exc:
        return json.dumps(
            {"ok": False, "error": str(exc), "available": sorted(SERVICE_TABLE)},
            ensure_ascii=False,
        )
    try:
        data = get_json(url, timeout=10)
        return json.dumps({"ok": True, "service": name, "health": data}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {"ok": False, "service": name, "url": url, "error": f"{type(exc).__name__}: {exc}"},
            ensure_ascii=False,
        )
