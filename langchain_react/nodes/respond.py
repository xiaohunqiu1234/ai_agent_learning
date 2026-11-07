"""收口节点：把最后一轮模型输出整理成 final_answer。

注意 trace（Thought/Action/Observation 过程）只留在 state.trace 里供日志与 SSE 展示，
不进 messages —— 否则过程文本会污染上下文，几轮之后模型就开始胡说。
"""

from __future__ import annotations

from qc_agent_mcp.prompts.react import parse_text_action
from qc_agent_mcp.state import QCAgentState


def node(state: QCAgentState, config=None) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"final_answer": "（无输出）"}

    last = messages[-1]
    content = getattr(last, "content", "") or ""

    parsed = parse_text_action(content)
    if parsed.get("type") == "final":
        return {"final_answer": parsed["answer"]}

    if getattr(last, "tool_calls", None):
        return {
            "final_answer": content
            or "（已达思考上限仍未收口，请稍后重试或补充更具体的条件）"
        }

    return {"final_answer": content}
