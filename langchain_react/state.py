"""State 定义 —— 遵循 LangGraph 规范：TypedDict + Annotated reducer。

为什么不用 dataclass / pydantic model：
    Checkpointer 要能把 state 序列化进 SQLite，TypedDict 是最稳的形态。

两类数据分开存，这是本项目相对"纯 history 多轮"的核心增量：
    messages —— 对话正文（短期记忆载体，由 Checkpointer 按 thread_id 持久化）
    slots    —— 跨轮累积的结构化槽位（file_ids / session_id / job_id …）
                用户第二轮说"好了吗"，靠的就是上轮存进 slots 的 session_id。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class Slots(TypedDict, total=False):
    """跨轮累积的槽位（报告生成 / 审查 / 优化 共用）。"""

    # 7022 报告生成
    free_text: str
    file_ids: list[str]
    report_id: list[str]
    report_type: str
    session_id: str
    attachment_id: str

    # 7029 报告优化
    original_docx_id: str
    review_result_docx_id: str
    job_id: str

    # 7025 QC 审查
    task_id: str

    # 通用
    access_token: str


class QCAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: str                              # Generate_Report / Review_Report / ...
    slots: Slots                             # 无 reducer：节点返回即整体覆盖
    missing: list[str]                       # 本轮需向用户追问的槽位
    iterations: Annotated[int, add]           # ReAct 计数（替代 max_iterations）
    trace: Annotated[list[dict], add]         # 工具轨迹：只供日志/SSE，不进 messages
    final_answer: str
