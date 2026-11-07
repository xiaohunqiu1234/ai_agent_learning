"""统一调用入口 —— 按用户需求理解意图、规划工具链、依次调度执行。

定位：
    · 领域工具层（qc_agent_mcp/tools/*）是"原子能力"；
    · 本模块是"任务级入口"：一句话需求进来，规划出工具调用序列并执行到底，
      前序产出（task_id / job_id / session_id）自动注入后续调用；
    · 与 ReAct 图（graph.py reason⇄act）并行存在：图走自主规划，
      本入口走确定性规划，二者共用同一套工具与统一文件参数契约。

意图理解优先走 7028 意图判断服务，失败降级 route.detect_intent 规则。
"""

from __future__ import annotations

import json
import time

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from qc_agent_mcp.nodes.route import detect_intent, extract_slots
from qc_agent_mcp.utils.text import truncate


class UnifiedRequest(BaseModel):
    """统一入参：一句话需求 + 统一文件契约 + 可选槽位。"""

    query: str = Field(description="用户原始需求描述")
    file_paths: list[str] = Field(default=[], description="本地文件路径列表（与 file_ids 二选一）")
    file_ids: list[str] = Field(default=[], description="云附件ID列表（需 access_token）")
    access_token: str = Field(default="", description="云平台访问令牌")
    report_type: str = Field(default="", description="报告类型（创新型/问题解决型-指令性目标/问题解决型-自定目标）")
    free_text: str = Field(default="", description="报告生成的自由描述（选填）")
    project_category: str = Field(default="", description="项目分类（工具推荐用）")
    project_type: str = Field(default="", description="项目类型（工具推荐用）")
    activity_node_name: str = Field(default="", description="活动节点名称（工具推荐用）")
    user_id: str = Field(default="qc_agent", description="用户标识")
    conversation_id: str = Field(default="", description="对话ID（7028 多轮续接用）")
    wait_seconds: int = Field(default=1800, description="长任务等待上限（秒）")


# ── 意图理解 ──────────────────────────────────────────────────────────

_INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    # 顺序对齐 route.py：优化在审查前，避免"根据审查意见优化"被误判
    ("Optimize_Report", ("优化报告", "润色报告", "修订报告", "按意见修改", "根据审查意见", "按审查意见", "优化")),
    ("Generate_Report", ("生成报告", "撰写报告", "写报告", "起草报告", "出报告", "生成")),
    ("Review_Report", ("审查", "审核", "合规检查", "挑错", "找错", "把关")),
    ("Tool_Recommend", ("推荐工具", "用什么工具", "管理工具推荐", "适用工具", "推荐")),
]


def _map_intent_text(text: str) -> str:
    """把 7028 返回的自然语言意图映射成内部意图名；映射不出返回空串。"""
    for intent, kws in _INTENT_KEYWORDS:
        if any(k in text for k in kws):
            return intent
    return ""


def understand_intent(req: UnifiedRequest) -> tuple[str, str]:
    """返回 (intent, source)。7028 优先，失败降级规则。"""
    from qc_agent_mcp.tools.intent import call_intent_judgment

    try:
        raw = call_intent_judgment(req.query, req.conversation_id, req.user_id)
        answer = str(raw.get("answer") or "")
        mapped = _map_intent_text(answer)
        if mapped:
            return mapped, "7028"
    except Exception:  # noqa: BLE001 —— 意图服务不可用就降级，不阻断
        pass
    return detect_intent(req.query), "rules"


# ── 规划 & 调度 ───────────────────────────────────────────────────────

def _invoke(tool_name: str, args: dict) -> tuple[str, dict]:
    """调用领域工具并解析 JSON 输出。返回 (结果文本, 结果dict或空)。

    find_tool 延迟导入：registry 在模块层 import 本文件（unified_qc_task），
    模块层再反向 import registry 会构成循环导入。
    """
    from qc_agent_mcp.tools.registry import find_tool

    tool = find_tool(tool_name)
    if tool is None:
        return json.dumps({"ok": False, "error": f"工具未注册: {tool_name}"}, ensure_ascii=False), {}
    out = tool.invoke(args)
    if isinstance(out, (dict, list)):
        out = json.dumps(out, ensure_ascii=False)
    try:
        return str(out), json.loads(out)
    except (json.JSONDecodeError, TypeError):
        return str(out), {}


def _has_files(req: UnifiedRequest) -> bool:
    return bool(req.file_paths or req.file_ids)


def _missing_files_msg(task: str) -> str:
    return json.dumps({
        "ok": False,
        "error": f"执行 {task} 需要文件：请提供 file_paths（本地路径）或 file_ids（附件ID，需 access_token）",
    }, ensure_ascii=False)


def _run_generate_report(req: UnifiedRequest, steps: list) -> str:
    if not _has_files(req) and not (req.free_text or req.query):
        return _missing_files_msg("报告生成")
    text, data = _invoke("generate_report", {
        "free_text": req.free_text or req.query,
        "report_type": req.report_type,
        "file_paths": req.file_paths,
        "file_ids": req.file_ids,
        "access_token": req.access_token,
    })
    steps.append({"tool": "generate_report", "result": data or text})
    return text


def _run_review_report(req: UnifiedRequest, steps: list) -> str:
    if not _has_files(req):
        return _missing_files_msg("报告审查")
    text, data = _invoke("qc_review_upload", {
        "query": req.free_text or "开始审查",
        "file_paths": req.file_paths,
        "file_ids": req.file_ids,
        "access_token": req.access_token,
        "user_id": req.user_id,
        "report_type": req.report_type,
    })
    steps.append({"tool": "qc_review_upload", "result": data or text})
    return text


def _run_optimize_report(req: UnifiedRequest, steps: list) -> str:
    # 优化 = 提交（原始+审查两份文档）→ 轮询 → 下载
    if not _has_files(req):
        return _missing_files_msg("报告优化（需原始报告 + 审查报告两份文档）")
    # 文件分配：第 1 份 → 原始，第 2 份 → 审查（本地/附件混传由工具层支持）
    text, data = _invoke("submit_optimize_job", {
        "original_file_paths": req.file_paths[:1], "original_file_ids": req.file_ids[:1],
        "review_file_paths": req.file_paths[1:2], "review_file_ids": req.file_ids[1:2],
        "access_token": req.access_token, "report_title": req.report_type,
    })
    steps.append({"tool": "submit_optimize_job", "result": data or text})
    job_id = data.get("job_id") if isinstance(data, dict) else ""
    if not job_id:
        return text  # 提交失败，直接返回错误

    # 轮询到终态
    deadline = time.time() + max(60, min(req.wait_seconds, 3600))
    status_text, status_data = "", {}
    while time.time() < deadline:
        status_text, status_data = _invoke("get_optimize_status", {"job_id": job_id})
        st = status_data.get("status", "") if isinstance(status_data, dict) else ""
        steps.append({"tool": "get_optimize_status", "result": status_data})
        if st in ("done", "failed", "error"):
            break
        time.sleep(5)
    if isinstance(status_data, dict) and status_data.get("status") == "done":
        dl_text, dl_data = _invoke("download_optimized_report", {"job_id": job_id})
        steps.append({"tool": "download_optimized_report", "result": dl_data or dl_text})
        return dl_text
    return status_text or text


def _run_tool_recommend(req: UnifiedRequest, steps: list) -> str:
    if not (req.activity_node_name or req.project_category or req.project_type):
        return json.dumps({
            "ok": False,
            "error": "工具推荐需要项目信息：请提供 activity_node_name（活动节点），"
                     "以及 project_category / project_type",
        }, ensure_ascii=False)
    text, data = _invoke("recommend_ai", {
        "project_category": req.project_category,
        "project_type": req.project_type,
        "activity_node_name": req.activity_node_name,
    })
    steps.append({"tool": "recommend_ai", "result": data or text})
    return text


def _run_kb_qa(req: UnifiedRequest, steps: list) -> str:
    text, data = _invoke("general_qa", {
        "query": req.query,
        "file_paths": req.file_paths,
        "file_ids": req.file_ids,
        "access_token": req.access_token,
        "user_id": req.user_id,
    })
    steps.append({"tool": "general_qa", "result": data or text})
    return text


def _run_task_status(req: UnifiedRequest, steps: list) -> str:
    """从用户文本抽任务指针（session_id / job_id / task_id）→ 转发到对应查询工具。"""
    slots = extract_slots(req.query)

    def _first(v):
        return v[0] if isinstance(v, list) else v

    if slots.get("job_id"):
        tool_name, args = "get_optimize_status", {"job_id": _first(slots["job_id"])}
    elif slots.get("task_id"):
        tool_name, args = "qc_review_task_status", {"task_id": _first(slots["task_id"])}
    elif slots.get("session_id"):
        tool_name, args = "get_report_attachment", {"session_id": _first(slots["session_id"])}
    else:
        return json.dumps(
            {"ok": False, "error": "查询进度需要任务号：请提供 session_id / job_id / task_id"},
            ensure_ascii=False)
    text, data = _invoke(tool_name, args)
    steps.append({"tool": tool_name, "result": data or text})
    return text


_PLANNERS = {
    "Generate_Report": _run_generate_report,
    "Review_Report": _run_review_report,
    "Optimize_Report": _run_optimize_report,
    "Tool_Recommend": _run_tool_recommend,
    "KB_QA": _run_kb_qa,
    "Task_Status": _run_task_status,
}


def run_unified_task(req: UnifiedRequest) -> str:
    """统一调用入口：理解意图 → 规划工具链 → 依次调度 → 汇总返回（JSON 字符串）。"""
    steps: list[dict] = []
    try:
        intent, source = understand_intent(req)
        planner = _PLANNERS.get(intent)
        if planner is None:
            intent, source = detect_intent(req.query), "rules_fallback"
            planner = _PLANNERS.get(intent)
        result = (planner(req, steps) if planner else json.dumps(
            {"ok": False, "error": f"无法识别的任务类型: {intent}"}, ensure_ascii=False))
        return json.dumps({
            "ok": True,
            "intent": intent,
            "intent_source": source,
            "steps": steps,
            "result": truncate(result, 6000),
        }, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 —— 入口绝不抛异常给调用方
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}", "steps": steps},
                          ensure_ascii=False)


@tool
def unified_qc_task(
    query: str = "",
    file_paths: list[str] = [],
    file_ids: list[str] = [],
    access_token: str = "",
    report_type: str = "",
    project_category: str = "",
    project_type: str = "",
    activity_node_name: str = "",
) -> str:
    """QC 任务统一入口：一句话需求 → 自动理解意图、规划工具链并执行到底（长任务）。

    适合"帮我审查这份报告 / 生成一份报告 / 优化报告 / 推荐工具"这类端到端任务。
    文件双模式：file_paths（本地路径）或 file_ids（附件ID + access_token），二选一。
    返回 JSON：intent（识别的意图）、steps（每步工具产出）、result（最终结果）。
    如果只需要单个原子操作（如查状态、知识问答），优先用对应的领域工具。
    """
    return run_unified_task(UnifiedRequest(
        query=query,
        file_paths=file_paths or [],
        file_ids=file_ids or [],
        access_token=access_token,
        report_type=report_type,
        project_category=project_category,
        project_type=project_type,
        activity_node_name=activity_node_name,
    ))
