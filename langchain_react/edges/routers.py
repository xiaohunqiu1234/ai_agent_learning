"""路由函数：决定图下一步走哪个节点。

should_continue 就是 ReAct 循环的出口判定，等价于 AgentExecutor 内部的
while 循环判断，但在这里是显式可见、可改、可单测的。
"""

from __future__ import annotations

from langgraph.graph import END

from qc_agent_mcp.configuration import Configuration
from qc_agent_mcp.prompts.react import parse_text_action
from qc_agent_mcp.state import QCAgentState


def route_by_intent(state: QCAgentState) -> str:
    """槽位不全先追问，齐全才进入思考链路。"""
    return "clarify" if state.get("missing") else "reason"


def should_continue(state: QCAgentState, config=None) -> str:
    """ReAct 循环出口：
        act      —— 还要调工具（含"格式错误需纠正"的情况）
        respond  —— 已给出 Final Answer
        END      —— 达到迭代上限，强制收口
    """
    cfg = Configuration.from_runnable_config(config)
    if state.get("iterations", 0) >= cfg.max_iterations:
        return "respond"

    messages = state.get("messages", [])
    if not messages:
        return "respond"

    last = messages[-1]

    if getattr(last, "tool_calls", None):
        return "act"

    parsed = parse_text_action(getattr(last, "content", "") or "")
    ptype = parsed.get("type")
    if ptype == "action":
        return "act"
    if ptype == "final":
        return "respond"

    # 格式不对：交给 act 回灌纠正提示，再由 act→reason 回到这里重试
    return "act"


def to_end(state: QCAgentState) -> str:  # noqa: ARG001
    """显式终点（供需要占位边的场景使用）。"""
    return END
