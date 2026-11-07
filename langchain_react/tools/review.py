"""7025 QC 审查服务 → LangChain 工具。

接口：
    POST /qc/ai/qc_review/tasks/upload   —— 上传文件并启动审查（multipart + SSE 长流）
    GET  /qc/ai/qc_review/tasks          —— 任务列表
    GET  /qc/ai/qc_review/tasks/{task_id} —— 单任务详情（含 7015 实时状态）

上传双模式（对齐服务端）：
    附件ID模式  → use_attachment=true + attachmentId + access_token
    文件流模式  → docx_file（单文件）
SSE 终态：answer（审查意见全文）→ completed（identified_type/total_issues/…）；
失败：status=failed + error。工具内部消费完整流。
"""

from __future__ import annotations

import json

from langchain_core.tools import tool

from qc_agent_mcp.services import service_url
from qc_agent_mcp.tools.file_resolver import resolve_files, multipart_files
from qc_agent_mcp.tools.http_client import get_json, safe_call, short
from qc_agent_mcp.tools.stream import compact_sse, sse_collect

_SSE_TIMEOUT = 1800.0  # 审查是分钟级长任务


@tool
def qc_review_upload(
    query: str = "开始审查",
    file_paths: list[str] = [],
    file_ids: list[str] = [],
    access_token: str = "",
    user_id: str = "qc_agent",
    full_text_review: str = "false",
    report_type: str = "",
) -> str:
    """上传 QC 报告并启动智能审查（7025，长任务，同步等待审查完成）。

    双模式入参（二选一）：file_paths=[本地报告路径]（单文件）或
    file_ids=[云附件ID] + access_token。
    query 为审查指令；report_type 可指定报告类型（选填）。
    返回 JSON：task_id、审查意见 answer、identified_type、total_issues、cloud_attachment_id。
    """
    resolved, err = resolve_files({"file_paths": file_paths, "file_ids": file_ids,
                                   "access_token": access_token})
    if err:
        return json.dumps({"ok": False, "error": err}, ensure_ascii=False)

    url = service_url("qc_review", "/qc/ai/qc_review/tasks/upload")
    if resolved["mode"] == "ids":
        if len(resolved["file_ids"]) != 1:
            return json.dumps({"ok": False, "error": "审查仅支持单个文件：file_ids 只能传 1 个附件ID"},
                              ensure_ascii=False)
        fields = {
            "use_attachment": "true",
            "attachmentId": resolved["file_ids"][0],
            "query": query, "user_id": user_id,
            "full_text_review": full_text_review, "report_type": report_type,
        }
        files = None
    else:
        if len(resolved["files"]) != 1:
            return json.dumps({"ok": False, "error": "审查仅支持单个文件：file_paths 只能传 1 个本地路径"},
                              ensure_ascii=False)
        fields = {"query": query, "user_id": user_id,
                  "full_text_review": full_text_review, "report_type": report_type}
        name, data, mime = resolved["files"][0]
        files = [("docx_file", (name, data, mime))]

    sse = sse_collect(url, fields=fields, files=files,
                      access_token=access_token, timeout=_SSE_TIMEOUT)
    out = compact_sse(sse, answer_limit=4000)
    for ev in sse.get("events", []):
        if ev.get("id") and "task_id" not in out:
            out["task_id"] = ev["id"]
        if ev.get("cloud_attachment_id"):
            out["cloud_attachment_id"] = ev["cloud_attachment_id"]
        if ev.get("identified_type"):
            out["identified_type"] = ev["identified_type"]
        if ev.get("total_issues") is not None:
            out["total_issues"] = ev["total_issues"]
    fin = sse.get("final") or {}
    if fin.get("error"):
        out["ok"] = False
        out["error"] = fin["error"]
    return short(json.dumps(out, ensure_ascii=False), 6000)


@tool
def qc_review_task_status(task_id: str = "") -> str:
    """查询 QC 审查任务详情（7025，只读）。

    输入 qc_review_upload 返回的 task_id，返回任务状态、审查结果摘要与 7015 实时进度。
    用户问"审查好了吗/审查进度"时使用。
    """
    def _call():
        return get_json(
            service_url("qc_review", f"/qc/ai/qc_review/tasks/{task_id}"),
            timeout=30,
        )
    return short(safe_call(_call), 3000)


@tool
def qc_review_list_tasks(user: str = "") -> str:
    """列出 QC 审查任务摘要（7025，只读，按创建时间倒序）。

    user 选填，传入则只返回该用户的任务。用户想找历史审查任务时使用。
    """
    def _call():
        return get_json(
            service_url("qc_review", "/qc/ai/qc_review/tasks"),
            timeout=30,
        )
    return short(safe_call(_call), 3000)
