"""7027 AI 评审意见服务 → LangChain 工具。

接口：POST /qc/ai/evaluation/api/review_comments（端口 7027）
请求体透传给核心编排方法 do_review_comments（下载 → 解析 → 评审全流程），
返回 (status_code, content)。评审可能耗时较长，超时给足。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from qc_agent_mcp.services import service_url
from qc_agent_mcp.tools.http_client import post_json, safe_call, short

_TIMEOUT = 1800.0  # 评审含下载解析与 LLM 评审，按长任务给超时


@tool
def ai_review_comments(payload_json: str = "{}") -> str:
    """AI 评审报告并返回评审意见（7027，长任务）。

    输入完整请求体 JSON 字符串（透传给评审编排核心，通常包含报告附件信息/评审参数）。
    返回评审意见内容。仅在用户明确要求"评审/评审意见"时使用。
    """
    def _call():
        body = json.loads(payload_json or "{}")
        return post_json(
            service_url("ai_review", "/qc/ai/evaluation/api/review_comments"),
            payload=body,
            timeout=_TIMEOUT,
        )
    return short(safe_call(_call), 4000)
