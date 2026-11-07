"""图装配 —— 必须导出名为 graph 的编译后对象（langgraph.json 的加载契约）。

P1 拓扑：

    START → route ─┬─ clarify（槽位不全，追问后结束本轮）
                   └─ reason ⇄ act → respond → END

`act → reason` 这条边就是 ReAct 循环；迭代上限由 configuration.max_iterations 控制。
P3 会在此处按业务域挂子图（检索/生成/审查/优化），拓扑不变、只加分支。
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from qc_agent_mcp.edges.routers import route_by_intent, should_continue
from qc_agent_mcp.memory.checkpointer import build_checkpointer
from qc_agent_mcp.memory.store import build_store
from qc_agent_mcp.nodes import act, clarify, reason, respond, route
from qc_agent_mcp.state import QCAgentState

builder = StateGraph(QCAgentState)

builder.add_node("route", route.node)
builder.add_node("clarify", clarify.node)
builder.add_node("reason", reason.node)
builder.add_node("act", act.node)
builder.add_node("respond", respond.node)

builder.add_edge(START, "route")
builder.add_conditional_edges(
    "route",
    route_by_intent,
    {"clarify": "clarify", "reason": "reason"},
)
builder.add_edge("clarify", END)

builder.add_conditional_edges(
    "reason",
    should_continue,
    {"act": "act", "respond": "respond", "reason": "reason"},
)
builder.add_edge("act", "reason")
builder.add_edge("respond", END)

# 短期记忆：thread_id(=conversation_id) 级；长期记忆：跨会话，P3 启用
graph = builder.compile(
    checkpointer=build_checkpointer(),
    store=build_store(),
    name="qc_agent",
)

__all__ = ["builder", "graph"]
