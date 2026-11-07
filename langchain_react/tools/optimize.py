"""7029 QC 报告优化服务 → LangChain 工具（submit / poll / download 三步走）。

接口：
    POST /qc/ai/optimize          —— 提交优化任务（multipart，异步，返回 job_id）
    GET  /qc/ai/status/{job_id}   —— 查询任务状态
    GET  /qc/ai/download/{job_id} —— 下载优化后 docx（二进制）

优化必须同时提供"原始报告"与"审查报告"两份文档，各自独立支持
本地路径 / 附件ID（允许混传，对齐服务端）。download 是二进制文件，
不能塞进 Observation，所以落盘后返回路径。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from qc_agent_mcp.services import service_url
from qc_agent_mcp.tools.file_resolver import resolve_files
from qc_agent_mcp.tools.http_client import get_json, safe_call, short, TRUST_ENV

_TIMEOUT = 300.0


class _OptimizeSubmit(BaseModel):
    """原始报告 + 审查报告 双模式提交参数（允许本地/附件混传）。"""

    original_file_paths: list[str] = Field(default=[], description="原始报告本地路径（与 original_file_ids 二选一）")
    original_file_ids: list[str] = Field(default=[], description="原始报告云附件ID（需 access_token）")
    review_file_paths: list[str] = Field(default=[], description="审查报告本地路径（与 review_file_ids 二选一）")
    review_file_ids: list[str] = Field(default=[], description="审查报告云附件ID（需 access_token）")
    access_token: str = Field(default="", description="云平台访问令牌，附件ID模式必填")
    report_title: str = Field(default="", description="优化输出报告标题（选填）")


def _one_file(paths: list[str], label: str):
    """单文件校验 + 读取，返回 (name, bytes, mime) 或 (None, 错误文本)。"""
    resolved, err = resolve_files({"file_paths": paths})
    if err:
        return None, f"{label}: {err}"
    if len(resolved["files"]) != 1:
        return None, f"{label}: 只能提供 1 份文件，收到 {len(resolved['files'])} 份"
    return resolved["files"][0], None


@tool
def submit_optimize_job(
    original_file_paths: list[str] = [],
    original_file_ids: list[str] = [],
    review_file_paths: list[str] = [],
    review_file_ids: list[str] = [],
    access_token: str = "",
    report_title: str = "",
) -> str:
    """提交报告优化任务（7029，异步，立即返回 job_id）。

    必须提供两份文档（本地路径或附件ID，可混传）：
      · 原始报告：original_file_paths / original_file_ids
      · 审查报告：review_file_paths / review_file_ids（通常来自 qc_review_upload 的产出）
    附件ID模式需 access_token。返回 JSON：job_id（用于 get_optimize_status 轮询）。
    """
    p = _OptimizeSubmit(
        original_file_paths=original_file_paths or [],
        original_file_ids=original_file_ids or [],
        review_file_paths=review_file_paths or [],
        review_file_ids=review_file_ids or [],
        access_token=access_token,
        report_title=report_title,
    )

    fields: dict = {"report_title": p.report_title}
    files: list[tuple] = []

    # 原始报告：路径模式 → original_docx 文件流；附件模式 → original_docx_id
    if p.original_file_paths:
        f, err = _one_file(p.original_file_paths, "原始报告")
        if err:
            return json.dumps({"ok": False, "error": err}, ensure_ascii=False)
        files.append(("original_docx", f))
    elif p.original_file_ids:
        if not p.access_token:
            return json.dumps({"ok": False, "error": "原始报告附件ID模式需要 access_token"}, ensure_ascii=False)
        fields["original_docx_id"] = p.original_file_ids[0]
    else:
        return json.dumps(
            {"ok": False, "error": "缺少原始报告：请提供 original_file_paths 或 original_file_ids"},
            ensure_ascii=False)

    # 审查报告：路径模式 → review_result_docx；附件模式 → review_result_docx_id
    if p.review_file_paths:
        f, err = _one_file(p.review_file_paths, "审查报告")
        if err:
            return json.dumps({"ok": False, "error": err}, ensure_ascii=False)
        files.append(("review_result_docx", f))
    elif p.review_file_ids:
        if not p.access_token:
            return json.dumps({"ok": False, "error": "审查报告附件ID模式需要 access_token"}, ensure_ascii=False)
        fields["review_result_docx_id"] = p.review_file_ids[0]
    else:
        return json.dumps(
            {"ok": False, "error": "缺少审查报告：请提供 review_file_paths 或 review_file_ids"},
            ensure_ascii=False)

    try:
        with httpx.Client(timeout=_TIMEOUT, verify=False, trust_env=TRUST_ENV) as client:
            resp = client.post(
                service_url("report_optimization", "/qc/ai/optimize"),
                data={k: v for k, v in fields.items() if v not in (None, "")},
                files=files or None,
            )
            resp.raise_for_status()
            return short(json.dumps(resp.json(), ensure_ascii=False), 2000)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


@tool
def get_optimize_status(job_id: str = "") -> str:
    """查询报告优化任务状态（7029，只读）。

    输入 submit_optimize_job 返回的 job_id，返回 status（running/done/…）、
    message 进度与产物文件名。status=done 后可用 download_optimized_report 下载。
    """
    def _call():
        return get_json(
            service_url("report_optimization", f"/qc/ai/status/{job_id}"),
            timeout=30,
        )
    return short(safe_call(_call), 2000)


@tool
def download_optimized_report(job_id: str = "", save_dir: str = "/tmp/qc_optimized") -> str:
    """下载优化完成的报告 docx 到本地（7029）。

    输入 job_id；任务必须已完成（status=done）。文件保存到 save_dir，
    返回 JSON：保存路径 path 与文件名 file_name。请把路径告知用户。
    """
    def _call():
        url = service_url("report_optimization", f"/qc/ai/download/{job_id}")
        with httpx.Client(timeout=300.0, verify=False, trust_env=TRUST_ENV,
                          follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

            # 文件名：优先 Content-Disposition，兜底 job_id.docx
            cd = resp.headers.get("content-disposition", "")
            m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";\n]+)", cd)
            file_name = m.group(1).strip('"\'') if m else f"{job_id}.docx"

            out_dir = Path(save_dir).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / file_name
            out_path.write_bytes(resp.content)
            return {
                "ok": True,
                "path": str(out_path),
                "file_name": file_name,
                "size_bytes": len(resp.content),
            }
    return short(safe_call(_call), 2000)
