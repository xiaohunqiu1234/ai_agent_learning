"""ReAct 提示词与文本解析。

走 OpenAI tools 协议时不需要格式约束（服务端按 schema 产出参数）；
**不支持 tools 时才需要**这套文本格式——模型自己在正文里写
Thought / Action / Action Input，由本模块解析成工具调用。
"""

from __future__ import annotations

import json
import re
from typing import Any

# ── 文本 ReAct 提示词（不支持 tools 协议时使用）───────────────────────
REACT_TEXT_TEMPLATE = """{system}

你可以使用以下工具：
{tools}

工具名必须是 [{tool_names}] 之一。

请严格按下面格式输出（每次只输出一轮）：

Thought: 你正在思考下一步该做什么
Action: 要调用的工具名
Action Input: 传给该工具的参数，必须是合法 JSON；无参数就写 {{}}

如果你已经能回答用户，改用：
Thought: 我现在知道最终答案了
Final Answer: 对原始问题的最终答复

规则：
- 一次只输出一个 Action，等 Observation 回来再决定下一步。
- Action Input 必须是合法 JSON，键名与工具参数一致。
- 不要编造 Observation。

现在开始。
"""

# ── 格式纠错提示（等价于 AgentExecutor 的 handle_parsing_errors）───────
FORMAT_HINT = (
    "你的输出不符合规定格式，请重新输出，严格使用：\n"
    "Thought: 思考内容\n"
    "Action: 工具名（必须是给定工具之一）\n"
    "Action Input: 合法 JSON 参数\n"
    "或直接输出：Final Answer: 最终答复"
)

_ACTION_RE = re.compile(r"Action\s*:\s*(?P<name>[A-Za-z0-9_\-]+)")
_INPUT_RE = re.compile(r"Action\s*Input\s*:\s*(?P<arg>.*)", re.DOTALL)
_FINAL_RE = re.compile(r"Final\s*Answer\s*:\s*(?P<ans>.*)", re.DOTALL)


def parse_text_action(text: str) -> dict[str, Any]:
    """从模型正文里抠出 Action / Final Answer。

    返回：
        {"type": "action", "name": ..., "args": {...}}
        {"type": "final",  "answer": ...}
        {"type": "none"}
    """
    if not text:
        return {"type": "none"}

    m_final = _FINAL_RE.search(text)
    m_action = _ACTION_RE.search(text)

    # Final Answer 出现在最后一个 Action 之后 → 视为收口
    if m_final and (m_action is None or m_final.start() > m_action.start()):
        return {"type": "final", "answer": m_final.group("ans").strip()}

    if m_action:
        name = m_action.group("name").strip()
        m_in = _INPUT_RE.search(text)
        raw = m_in.group("arg").strip() if m_in else ""
        return {"type": "action", "name": name, "args": _parse_args(raw)}

    return {"type": "none"}


def _parse_args(raw: str) -> dict[str, Any]:
    """Action Input 解析：先当 JSON，失败则退化成 {"query": 原文}。"""
    if not raw or raw in ("None", "none", "null"):
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"query": parsed}
    except json.JSONDecodeError:
        return {"query": raw.strip().strip('"')}


def build_react_system(system: str, tools_brief: str, tool_names_str: str) -> str:
    return REACT_TEXT_TEMPLATE.format(
        system=system, tools=tools_brief, tool_names=tool_names_str
    )
