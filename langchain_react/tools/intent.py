"""7028 意图判断服务 → LangChain 工具。

接口：POST /qc/ai/intent/chat-messages
输入：{"inputs": {}, "query": ..., "user": ..., "conversation_id": ...}
输出：{"answer": ..., "conversation_id": ..., ...}（服务端非流式 JSON）
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from qc_agent_mcp.services import service_url
from qc_agent_mcp.tools.http_client import post_json, safe_call, short

_TIMEOUT = 60.0


def call_intent_judgment(query: str, conversation_id: str = "", user: str = "qc_agent") -> dict:
    """底层调用（供统一入口 entry.py 复用，不走 @tool 包装）。"""
    return post_json(
        service_url("intent_judgment", "/qc/ai/intent/chat-messages"),
        payload={"inputs": {}, "query": query, "user": user, "conversation_id": conversation_id},
        timeout=_TIMEOUT,
    )


@tool
def intent_judgment(query: str = "", conversation_id: str = "", user: str = "qc_agent") -> str:
    """判断用户意图（7028 意图判断服务）。

    输入用户的自然语言问题，返回意图分类结果（如：生成报告/审查报告/优化报告/工具推荐/知识问答）。
    当需要确定"用户到底想做什么"时，首先使用本工具。
    """
    def _call():
        return call_intent_judgment(query, conversation_id, user)
    return short(safe_call(_call), 3000)


@tool
def intent_judgment_raw(body_json: str = "{}") -> str:
    """向 7028 意图判断服务提交完整请求体（JSON 字符串透传）。

    请求体格式：{"inputs": {}, "query": "...", "user": "...", "conversation_id": "..."}
    仅在需要自定义请求体（如多轮对话续接）时使用；常规场景请用 intent_judgment。
    """
    def _call():
        body = json.loads(body_json or "{}")
        return post_json(
            service_url("intent_judgment", "/qc/ai/intent/chat-messages"),
            payload=body,
            timeout=_TIMEOUT,
        )
    return short(safe_call(_call), 3000)
