"""统一 HTTP 封装。

设计要点：
    · 所有异常都转成字符串返回（safe_call），绝不把堆栈抛给 Agent；
      抛异常会中断图，返回错误文本则只是一次失败的 Observation。
    · verify=False + 关闭告警：内网代理多为自签证书（与现有服务保持一致）。
"""

from __future__ import annotations

import json
import os
import urllib3
from typing import Any

import httpx

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_TIMEOUT = 30.0

# 内网服务必须绕过环境代理：开发机上常有 http_proxy=127.0.0.1:xxxx，
# httpx 默认 trust_env=True 会把 localhost:70xx 也发给代理，结果是 502。
# 确需代理时设 QC_HTTP_TRUST_ENV=1。
TRUST_ENV = os.getenv("QC_HTTP_TRUST_ENV", "0") == "1"


def _headers(access_token: str = "", extra: dict | None = None) -> dict:
    h = {"Content-Type": "application/json"}
    if access_token:
        h["access-token"] = access_token
    if extra:
        h.update(extra)
    return h


def request_json(
    method: str,
    url: str,
    *,
    payload: dict | None = None,
    access_token: str = "",
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict | None = None,
) -> dict:
    """JSON 请求，返回 dict；非 JSON 响应包装成 {"raw": ...}。"""
    with httpx.Client(timeout=timeout, verify=False, trust_env=TRUST_ENV) as client:
        resp = client.request(
            method, url, json=payload, headers=_headers(access_token, headers)
        )
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"raw": resp.text}


def get_json(url: str, **kwargs) -> dict:
    return request_json("GET", url, **kwargs)


def post_json(url: str, payload: dict | None = None, **kwargs) -> dict:
    return request_json("POST", url, payload=payload, **kwargs)


def post_form(
    url: str,
    *,
    fields: dict | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    access_token: str = "",
    timeout: float = 120.0,
) -> dict:
    """multipart 表单提交（7022/7026/7025 都是这种形态）。

    files 元素格式：("files", ("a.docx", b"...", "application/octet-stream"))
    """
    data = {k: v for k, v in (fields or {}).items() if v not in (None, "")}
    if access_token:
        data["access_token"] = access_token

    with httpx.Client(timeout=timeout, verify=False, trust_env=TRUST_ENV) as client:
        resp = client.post(url, data=data, files=files or None,
                           headers={"access-token": access_token} if access_token else None)
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"raw": resp.text}


def safe_call(fn, *args, **kwargs) -> str:
    """执行任意调用并把结果 JSON 化；异常转成错误文本，不抛。"""
    try:
        result = fn(*args, **kwargs)
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
    except Exception as exc:  # noqa: BLE001 —— 各服务报错类型不一致，统一兜住
        return json.dumps(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
        )


def short(text: Any, limit: int = 3000) -> str:
    """截断任意结果为字符串，供 Observation 回灌。"""
    from qc_agent_mcp.utils.text import truncate

    if isinstance(text, (dict, list)):
        text = json.dumps(text, ensure_ascii=False)
    return truncate(str(text), limit)
