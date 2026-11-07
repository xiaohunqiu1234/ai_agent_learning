"""7026 常规问答服务 → LangChain 工具。

接口：POST /qc/ai/general_chat/chat/upload（multipart + SSE 长流）
    字段：query / user_id / access_token / cloud_obj_id
    文件双模式：files（多文件流）或 file_ids（附件ID，需 access_token），禁止混用
SSE 事件：start → answer_chunk(text=…)（逐段拼接）→ …。
工具内部消费完整流，拼接出完整回答。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from qc_agent_mcp.services import service_url
from qc_agent_mcp.tools.file_resolver import resolve_files, multipart_files
from qc_agent_mcp.tools.http_client import short
from qc_agent_mcp.tools.stream import compact_sse, sse_collect

_SSE_TIMEOUT = 600.0


@tool
def general_qa(
    query: str = "",
    file_paths: list[str] = [],
    file_ids: list[str] = [],
    access_token: str = "",
    cloud_obj_id: str = "",
    user_id: str = "qc_agent",
) -> str:
    """QC 知识问答 / 带文件的通用问答（7026）。

    输入用户问题 query；可选文件：file_paths（本地路径，可多份）或
    file_ids（附件ID，需 access_token），二选一。
    返回 JSON：完整回答 answer 与 conversation_id。
    适合：QC 概念咨询、报告规范问答、基于上传文档内容的问答。
    """
    resolved, err = resolve_files({"file_paths": file_paths, "file_ids": file_ids,
                                   "access_token": access_token}, allow_empty=True)
    if err:
        return json.dumps({"ok": False, "error": err}, ensure_ascii=False)

    fields = {"query": query, "user_id": user_id, "cloud_obj_id": cloud_obj_id}
    if resolved["mode"] == "path":
        files = multipart_files(resolved, "files")
    elif resolved["mode"] == "ids":
        files = None
        fields["file_ids"] = resolved["file_ids"]
    else:
        files = None

    sse = sse_collect(
        service_url("general_qa", "/qc/ai/general_chat/chat/upload"),
        fields=fields, files=files, access_token=access_token, timeout=_SSE_TIMEOUT,
    )
    out = compact_sse(sse, answer_limit=4000)
    for ev in sse.get("events", []):
        if ev.get("conversation_id"):
            out["conversation_id"] = ev["conversation_id"]
    return short(json.dumps(out, ensure_ascii=False), 6000)
