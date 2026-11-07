"""统一文件入参解析 —— 本地路径 / 云附件ID 双模式。

所有文件类工具（7022 报告生成 / 7025 审查 / 7026 文件问答 / 7029 优化）
共用本模块，Agent 侧只看到统一契约：

    file_paths : 本地文件路径列表（Agent 进程能读到的路径，读 bytes 转表单）
    file_ids   : 云附件 ID 列表（走 7031 附件模式，需 access_token）
    access_token: 附件模式必填

对齐各服务的约束：
    · 7026 明确禁止文件流与附件ID同时传 → 这里统一禁止混用；
    · 7025 附件模式必须 access_token；7022/7026 附件模式同样需要。

返回值约定：resolve_files() 成功返回 (resolved_dict, None)，失败返回 (None, 错误文本)；
错误文本直接当 Observation 回灌，不抛异常（与 tools/__init__.py 契约一致）。
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from pydantic import BaseModel, Field


class FileParams(BaseModel):
    """文件双模式统一入参（所有文件类工具的 args_schema 公共子集）。"""

    file_paths: list[str] = Field(default=[], description="本地文件路径列表（与 file_ids 二选一）")
    file_ids: list[str] = Field(default=[], description="云附件 ID 列表（与 file_paths 二选一）")
    access_token: str = Field(default="", description="云平台访问令牌，附件ID模式必填")


# 常见文档类型的 MIME 表（mimetypes 兜底）
_MIME_OVERRIDE = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _guess_mime(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext in _MIME_OVERRIDE:
        return _MIME_OVERRIDE[ext]
    return mimetypes.guess_type(name)[0] or "application/octet-stream"


def _as_list(value) -> list[str]:
    """模型偶尔把列表参数传成单个字符串或 JSON 数组字符串，这里统一归一。"""
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                return [str(x) for x in json.loads(s)]
            except json.JSONDecodeError:
                return [s]
        return [s]
    return [str(v) for v in value]


def normalize_params(params) -> FileParams:
    """把模型给的原始参数（dict / FileParams）归一成 FileParams。"""
    if isinstance(params, FileParams):
        return params
    params = params or {}
    return FileParams(
        file_paths=_as_list(params.get("file_paths")),
        file_ids=_as_list(params.get("file_ids")),
        access_token=str(params.get("access_token") or ""),
    )


def resolve_files(params, *, allow_empty: bool = False):
    """统一解析双模式文件参数。

    成功返回 (resolved, None)：
        {"mode": "path", "files": [(name, bytes, mime), ...]}
        {"mode": "ids",  "file_ids": [...]}
        {"mode": "none"}                          # allow_empty=True 且两者都为空
    失败返回 (None, 错误文本)。
    """
    p = normalize_params(params)

    if p.file_paths and p.file_ids:
        return None, "文件流与附件ID不能同时传：请统一使用 file_paths（本地路径）或 file_ids（附件ID）"

    if p.file_paths:
        files = []
        for fp in p.file_paths:
            path = Path(fp.strip())
            if not path.is_file():
                return None, f"本地文件不存在: {fp}"
            try:
                data = path.read_bytes()
            except OSError as exc:
                return None, f"读取文件失败: {fp} ({type(exc).__name__}: {exc})"
            files.append((path.name, data, _guess_mime(path.name)))
        return {"mode": "path", "files": files}, None

    if p.file_ids:
        if not p.access_token:
            return None, "附件ID模式需要 access_token（用于 7031 云文件中间件下载）"
        return {"mode": "ids", "file_ids": [x.strip() for x in p.file_ids if x.strip()]}, None

    if allow_empty:
        return {"mode": "none"}, None
    return None, "缺少文件：请提供 file_paths（本地路径）或 file_ids（附件ID，需 access_token）"


def multipart_files(resolved: dict, field: str = "files"):
    """把 resolve 的 path 模式结果转成 httpx multipart files 元组列表。

    元素格式：("files", ("a.docx", b"...", "application/..."))
    """
    if resolved.get("mode") != "path":
        return None
    return [(field, (name, data, mime)) for name, data, mime in resolved["files"]]
