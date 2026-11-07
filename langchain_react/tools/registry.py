"""工具注册表 —— 按副作用分组，控制"哪些工具能在这个会话里被调用"。

分组的意义：
    readonly     —— 只读，任何场景都可调用（检索、查状态、查健康）
    write        —— 有副作用（提交审查、提交优化），可在配置里整体关掉
    long_running —— 分钟级长任务（7022/7029/7025），必须走 submit + poll 两个工具

P1 只注册 readonly 的 system 工具；P2 起按域补充：
    kb.py（7026 知识检索问答）、report.py（7022）、review.py（7025）、
    optimize.py（7029）、tool_recommend.py（7001）、file.py（7031）
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from qc_agent_mcp.tools.system import current_time, service_health

# ── 7001 工具推荐（只读）──
from qc_agent_mcp.tools.tool_recommend import (
    enhance_table,
    recommend_ai,
    retrieve_tool_detail,
    tool_intent,
)
# ── 7028 意图判断（只读）──
from qc_agent_mcp.tools.intent import intent_judgment
# ── 7022 报告生成 ──
from qc_agent_mcp.tools.report import generate_report, get_report_attachment
# ── 7025 QC 审查 ──
from qc_agent_mcp.tools.review import qc_review_list_tasks, qc_review_task_status, qc_review_upload
# ── 7026 常规问答 ──
from qc_agent_mcp.tools.kb import general_qa
# ── 7027 AI 评审意见 ──
from qc_agent_mcp.tools.ai_review import ai_review_comments
# ── 7029 报告优化（异步任务）──
from qc_agent_mcp.tools.optimize import (
    download_optimized_report,
    get_optimize_status,
    submit_optimize_job,
)
# ── 统一入口工具（一句话需求 → 规划调度全链路）──
from qc_agent_mcp.entry import unified_qc_task

READONLY_TOOLS: list[BaseTool] = [
    current_time,
    service_health,
    # 意图理解
    intent_judgment,
    tool_intent,
    # 工具推荐
    recommend_ai,
    enhance_table,
    retrieve_tool_detail,
    # 任务状态查询
    qc_review_task_status,
    qc_review_list_tasks,
    get_optimize_status,
    get_report_attachment,
]

WRITE_TOOLS: list[BaseTool] = [
    ai_review_comments,      # 7027 评审
    general_qa,              # 7026 问答
    submit_optimize_job,     # 7029 提交优化
    download_optimized_report,  # 7029 下载产物
]

LONG_RUNNING_TOOLS: list[BaseTool] = [
    generate_report,         # 7022 分钟级
    qc_review_upload,        # 7025 分钟级
    unified_qc_task,         # 统一入口（内含长任务链）
]

ALL_TOOLS: list[BaseTool] = READONLY_TOOLS + WRITE_TOOLS + LONG_RUNNING_TOOLS


def get_tools(allow_write: bool = False) -> list[BaseTool]:
    """按开关返回本会话可用工具。"""
    return ALL_TOOLS if allow_write else list(READONLY_TOOLS)


def find_tool(name: str) -> BaseTool | None:
    """按名字找工具（文本 ReAct 路线靠它把 Action 名映射成真实工具）。"""
    target = (name or "").strip().lower()
    for t in ALL_TOOLS:
        if t.name.lower() == target:
            return t
    return None


def tool_brief() -> str:
    """生成工具清单文本，注入文本 ReAct 的提示词。"""
    lines = []
    for t in ALL_TOOLS:
        desc = (t.description or "").strip().splitlines()
        lines.append(f"- {t.name}: {desc[0] if desc else ''}")
    return "\n".join(lines)


def tool_names() -> list[str]:
    return [t.name for t in ALL_TOOLS]
