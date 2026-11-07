"""槽位补齐 / 追问澄清。

P1 形态：生成追问文案 → 直接结束本轮。用户下一轮回答后重新走 route，
此时 slots 已补齐（Checkpointer 持久化的），自然进入工具链路。

P4 升级点：改用 langgraph.types.interrupt() 做真正的 human-in-the-loop 中断，
上层通过 Command(resume=...) 续跑；届时对话与"任务内暂停"能统一处理。
"""

from __future__ import annotations

from qc_agent_mcp.nodes.route import question_for
from qc_agent_mcp.state import QCAgentState


def node(state: QCAgentState, config=None) -> dict:
    missing = state.get("missing") or []
    if not missing:
        return {}
    return {"final_answer": question_for(missing)}
