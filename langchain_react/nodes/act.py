"""工具执行节点：拿到 reason 的决策去跑工具，把结果作为 Observation 回灌。

两种产出形态必须分开，这是本节点最容易踩坑的地方：
    结构化路线 —— 回 ToolMessage（带 tool_call_id，OpenAI 协议要求）
    文本路线   —— 回 HumanMessage("Observation: …")
文本路线如果也用 ToolMessage，会因缺少对应的 tool_call_id 被服务端 400 拒绝。
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, ToolMessage

from qc_agent_mcp.configuration import Configuration
from qc_agent_mcp.prompts.react import FORMAT_HINT, parse_text_action
from qc_agent_mcp.state import QCAgentState
from qc_agent_mcp.tools.registry import find_tool, tool_names
from qc_agent_mcp.utils.text import truncate


def _coerce_args(tool, args: dict, raw: str) -> dict:
    """文本路线下模型常把整句话塞进参数里；单参数工具做一次兜底映射。"""
    schema = getattr(tool, "args_schema", None)
    fields = getattr(schema, "model_fields", None)
    if fields and len(fields) == 1 and "query" in args:
        only = next(iter(fields))
        if only != "query":
            return {only: args["query"]}
    return args


def _run(tool, args: dict, limit: int) -> tuple[str, bool]:
    try:
        out = tool.invoke(args)
        if isinstance(out, (dict, list)):
            out = json.dumps(out, ensure_ascii=False)
        return truncate(str(out), limit), True
    except Exception as exc:  # noqa: BLE001 —— 转成 Observation，不抛给图
        return json.dumps(
            {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False
        ), False


def node(state: QCAgentState, config=None) -> dict:
    cfg = Configuration.from_runnable_config(config)
    limit = cfg.observation_max_chars
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [HumanMessage(content=FORMAT_HINT)]}

    last = messages[-1]
    trace: list[dict] = []

    # ── 路线一：结构化 tool_calls ──────────────────────────────────────
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        out_msgs = []
        for tc in tool_calls:
            name = tc.get("name", "")
            args = tc.get("args") or {}
            tool = find_tool(name)
            if tool is None:
                content = json.dumps(
                    {"ok": False, "error": f"工具不存在: {name}", "available": tool_names()},
                    ensure_ascii=False,
                )
                ok = False
            else:
                content, ok = _run(tool, args, limit)
            out_msgs.append(ToolMessage(content=content, tool_call_id=tc.get("id") or name))
            trace.append({"tool": name, "args": args, "ok": ok, "chars": len(content)})
        return {"messages": out_msgs, "trace": trace}

    # ── 路线二：文本 ReAct ─────────────────────────────────────────────
    parsed = parse_text_action(getattr(last, "content", "") or "")
    if parsed.get("type") == "action":
        name = parsed["name"]
        tool = find_tool(name)
        if tool is None:
            content = json.dumps(
                {"ok": False, "error": f"工具不存在: {name}", "available": tool_names()},
                ensure_ascii=False,
            )
            trace.append({"tool": name, "args": parsed.get("args"), "ok": False})
        else:
            args = _coerce_args(tool, parsed.get("args") or {}, str(parsed.get("args")))
            content, ok = _run(tool, args, limit)
            trace.append({"tool": name, "args": args, "ok": ok, "chars": len(content)})
        return {"messages": [HumanMessage(content=f"Observation: {content}")], "trace": trace}

    # ── 路线三：格式不对 —— 给纠正提示，回 reason 重来 ──────────────────
    trace.append({"tool": None, "error": "格式错误，已回灌格式纠正提示"})
    return {"messages": [HumanMessage(content=FORMAT_HINT)], "trace": trace}
