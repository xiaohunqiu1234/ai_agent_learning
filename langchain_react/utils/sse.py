"""消费后端 SSE 接口（7022/7026 目前都是 SSE 返回）。

MCP 的 tools/call 是请求-响应模型，吃不下长连接流；
工具内部把流消费完再回灌成一段文本，是让旧服务能直接当工具用的关键适配层。
"""

from __future__ import annotations

import json
from typing import Iterator

import httpx

from qc_agent_mcp.tools.http_client import TRUST_ENV


def iter_sse(
    url: str,
    *,
    form: dict | list[tuple[str, str]] | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    json_body: dict | None = None,
    headers: dict | None = None,
    timeout: float = 300.0,
) -> Iterator[dict]:
    """逐条产出 SSE 的 data 载荷（已 json.loads；非 JSON 帧跳过）。

    form 与 files 同时给出时按 multipart 提交（7022/7026 都是这种形态）。
    form 传 list[tuple[str, str]] 时（需要重复字段名，如多个 file_ids）
    会自动转成 multipart 字段——httpx 的 data= 不支持重复字段名。
    """
    req_headers = headers or {}

    with httpx.Client(timeout=timeout, headers=req_headers, verify=False,
                      trust_env=TRUST_ENV) as client:
        # 请求级参数（data / files / json）必须在 stream() 上给，不能进 Client()
        if files:
            req: dict = {"data": form or {}, "files": files}
        elif isinstance(form, list):
            req = {"files": [(k, (None, v)) for k, v in form]}
        elif form is not None:
            req = {"data": form}
        else:
            req = {"json": json_body or {}}

        with client.stream("POST", url, **req) as resp:
            resp.raise_for_status()
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    frame, buf = buf.split("\n\n", 1)
                    for line in frame.splitlines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw in ("", "[DONE]"):
                            continue
                        try:
                            yield json.loads(raw)
                        except json.JSONDecodeError:
                            continue


def consume_sse_to_text(
    url: str,
    *,
    form: dict | list[tuple[str, str]] | None = None,
    files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
    json_body: dict | None = None,
    headers: dict | None = None,
    timeout: float = 300.0,
    content_keys: tuple[str, ...] = ("answer", "content", "progress", "message", "text"),
    max_chars: int = 6000,
) -> str:
    """把整条 SSE 流拼成一段文本（按出现顺序取第一个命中的字段）。"""
    parts: list[str] = []
    for ev in iter_sse(
        url, form=form, files=files, json_body=json_body, headers=headers, timeout=timeout
    ):
        for k in content_keys:
            v = ev.get(k)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
                break
        else:
            # 没有命中的文本字段时，把整帧保留（便于调试 completed/error 事件）
            parts.append(json.dumps(ev, ensure_ascii=False))
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n".join(parts)
