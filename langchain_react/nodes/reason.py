"""ReAct 思考节点：模型决定是否调用工具、调用哪个。

双路线（沿用 ReAct_local.py 最关键的判断，按模型实际能力切换）：
    支持 tools 协议 -> bind_tools，参数由服务端按 schema 产出并校验（稳）
    不支持         -> 文本 ReAct，靠提示词约束 Thought/Action/Action Input
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from qc_agent_mcp.configuration import Configuration
from qc_agent_mcp.prompts.react import build_react_system
from qc_agent_mcp.prompts.system import SYSTEM
from qc_agent_mcp.state import QCAgentState
from qc_agent_mcp.tools.registry import get_tools, tool_brief, tool_names
from qc_agent_mcp.utils.llm import build_llm, supports_tools


def node(state: QCAgentState, config=None) -> dict:
    cfg = Configuration.from_runnable_config(config)
    llm = build_llm(cfg)
    tools = get_tools()

    system = SYSTEM.format(
        slots=(state.get("slots") or {}),
    )

    if supports_tools(llm):
        llm = llm.bind_tools(tools)
    else:
        system = build_react_system(system, tool_brief(), ", ".join(tool_names()))

    messages = [SystemMessage(content=system), *state.get("messages", [])]
    ai = llm.invoke(messages)

    # 只返回增量：messages 由 add_messages 追加，iterations 由 add 累加
    return {"messages": [ai], "iterations": 1}
