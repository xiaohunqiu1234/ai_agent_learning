"""7022 QC 报告生成服务 → LangChain 工具。

接口：
    POST /qc/ai/report_generation/sse          —— SSE 长流生成报告（multipart 表单）
    GET  /qc/ai/report_generation/attachment/{session_id} —— 附件映射查询

SSE 事件：session（session_id）→ progress… → completed（cloud_attachment_id）/ status（失败）。
工具内部消费完整流，返回聚合后的最终结果。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from qc_agent_mcp.services import service_url
from qc_agent_mcp.tools.file_resolver import resolve_files, multipart_files
from qc_agent_mcp.tools.http_client import get_json, safe_call, short
from qc_agent_mcp.tools.stream import compact_sse, sse_collect

_SSE_TIMEOUT = 1800.0  # 报告生成是分钟级长任务


@tool
def generate_report(
    free_text: str = "",
    report_type: str = "",
    file_paths: list[str] = [],
    file_ids: list[str] = [],
    access_token: str = "",
) -> str:
    """生成 QC 报告（7022，长任务，同步等待生成完成）。

    双模式入参：file_paths（本地素材文件路径）或 file_ids（云附件ID，需 access_token），二选一；
    free_text 为用户自由描述（选填）；report_type 为中文报告类型
    （创新型 / 问题解决型-指令性目标 / 问题解决型-自定目标，选填，缺失由服务兜底）。
    返回 JSON：session_id、最终状态与 cloud_attachment_id（生成产物附件）。
    """
    resolved, err = resolve_files(
        {"file_paths": file_paths, "file_ids": file_ids, "access_token": access_token},
        allow_empty=True,
    )
    if err:
        return json.dumps({"ok": False, "error": err}, ensure_ascii=False)

    fields: dict = {"free_text": free_text, "report_type": report_type}
    if resolved["mode"] == "path":
        files = multipart_files(resolved, "files")
    elif resolved["mode"] == "ids":
        files = None
        fields["file_ids"] = resolved["file_ids"]
    else:
        files = None

    sse = sse_collect(
        service_url("report_generation", "/qc/ai/report_generation/sse"),
        fields=fields, files=files, access_token=access_token, timeout=_SSE_TIMEOUT,
    )
    out = compact_sse(sse, answer_limit=2000)
    # 提取关键产物字段：session_id / cloud_attachment_id
    for ev in sse.get("events", []):
        if ev.get("session_id") and "session_id" not in out:
            out["session_id"] = ev["session_id"]
        if ev.get("cloud_attachment_id"):
            out["cloud_attachment_id"] = ev["cloud_attachment_id"]
    fin = sse.get("final") or {}
    if fin.get("error"):
        out["ok"] = False
        out["error"] = fin["error"]
    return short(json.dumps(out, ensure_ascii=False), 4000)


@tool
def get_report_attachment(session_id: str = "") -> str:
    """查询报告生成任务的附件映射（7022，只读）。

    输入 generate_report 返回的 session_id，返回该任务的附件映射信息。
    """
    def _call():
        return get_json(
            service_url("report_generation", f"/qc/ai/report_generation/attachment/{session_id}"),
            timeout=30,
        )
    return short(safe_call(_call), 2000)
