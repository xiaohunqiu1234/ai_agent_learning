"""通用 SSE 聚合器 —— 7022/7025/7026 都是 text/event-stream 长流。

Agent 的 Observation 必须是"一次性的最终结果"，所以工具内部要把 SSE 流
完整消费完再返回。本模块做三件事：

    1. 以流式方式 POST（multipart 表单或 JSON），逐行解析 event:/data:；
    2. 遇到 data: [DONE] 或连接结束即停止；
    3. 聚合成统一结构：
         events   —— 全部事件（带 event 名）
         answer   —— 7026 answer_chunk.text / 7025 answer 字段的拼接
         final    —— 终态事件（completed / failed / status / message_end 等）
         progress —— 最后一条 progress 文本（失败排查用）

异常一律转成 {"ok": False, "error": ...} 返回，不抛给 Agent。
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from qc_agent_mcp.tools.http_client import TRUST_ENV, _headers

# 事件名里出现这些即视为"终态"
_TERMINAL_EVENTS = {"completed", "failed", "error", "status", "message_end", "end"}


def sse_collect(
    url: str,
    *,
    fields: dict | None = None,
    files: list[tuple] | None = None,
    payload: dict | None = None,
    access_token: str = "",
    timeout: float = 600.0,
    max_events: int = 4000,
) -> dict:
    """POST 并消费 SSE 流，返回聚合结果 dict（含 ok/error 兜底）。"""
    out: dict[str, Any] = {
        "ok": True,
        "events": [],
        "answer": "",
        "final": {},
        "progress": "",
    }
    try:
        with httpx.Client(timeout=timeout, verify=False, trust_env=TRUST_ENV) as client:
            headers = _headers(access_token)
            if payload is not None:
                resp = client.post(url, json=payload, headers=headers)
            else:
                data = {k: v for k, v in (fields or {}).items() if v not in (None, "")}
                if access_token:
                    data.setdefault("access_token", access_token)
                resp = client.post(url, data=data, files=files or None, headers=headers)
            resp.raise_for_status()

            current_event = "message"
            for raw_line in resp.iter_lines():
                line = (raw_line or "").strip()
                if not line:
                    current_event = "message"  # 空行 = 事件边界
                    continue
                if line.startswith(":"):
                    continue  # SSE 注释/保活行
                if line.startswith("event:"):
                    current_event = line[6:].strip() or "message"
                    continue
                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    data = {"raw": data_str}
                    out["answer"] += data_str  # 纯文本流兜底拼接

                ev = {"event": current_event, **data}
                if len(out["events"]) < max_events:
                    out["events"].append(ev)

                # 7026: answer_chunk 的 text 逐段拼接；7025: answer 一次性全文
                text = data.get("text")
                if isinstance(text, str) and current_event == "answer_chunk":
                    out["answer"] += text
                ans = data.get("answer")
                if isinstance(ans, str) and ans and current_event != "answer_chunk":
                    out["answer"] += ans

                prog = data.get("progress")
                if isinstance(prog, str) and prog:
                    out["progress"] = prog

                if current_event in _TERMINAL_EVENTS or any(
                    k in data for k in ("status", "error", "done")
                ):
                    out["final"] = ev
        return out
    except Exception as exc:  # noqa: BLE001 —— 流式调用任何异常都转结构化错误
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "events": out["events"],
            "answer": out["answer"],
            "final": out["final"],
            "progress": out["progress"],
        }


def compact_sse(sse: dict, *, answer_limit: int = 4000) -> dict:
    """把聚合结果压缩成 Observation 友好的小 dict：丢掉 events 明细，保留关键摘要。"""
    final = sse.get("final") or {}
    return {
        "ok": sse.get("ok", False),
        "error": sse.get("error", ""),
        "progress": sse.get("progress", ""),
        "answer": (sse.get("answer") or "")[:answer_limit],
        "final": final,
        "event_count": len(sse.get("events") or []),
    }
