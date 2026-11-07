"""意图路由 + 槽位识别。

P1 用规则实现（确定性、零 LLM 依赖、可单测）；
P2 可换成本地 LLM 或直接调用 7028 意图判断服务，接口不变。

槽位（slots）是本节点的核心产出：跨轮累积，写进 state 由 Checkpointer 持久化，
下一轮用户说"好了吗"时才能知道问的是哪个 session_id。
"""

from __future__ import annotations

import json
import re

from qc_agent_mcp.state import QCAgentState, Slots

# ── 意图规则：顺序敏感 ────────────────────────────────────────────────
# 1) Task_Status 排最前：用户说"生成报告进度"是在问进度，不是要新生成一份；
# 2) Optimize 排在 Review 前："根据审查意见优化报告" 同时含"审查"和"优化"。
INTENT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("Task_Status", ("进度", "好了吗", "好了没", "完成了吗", "任务状态", "查一下任务", "生成好了吗")),
    ("Optimize_Report", ("优化报告", "润色报告", "修订报告", "按意见修改", "根据审查意见", "按审查意见")),
    ("Generate_Report", ("生成报告", "撰写报告", "写一份报告", "起草报告", "出报告", "帮我写报告", "生成一份", "写报告")),
    ("Review_Report", ("审查报告", "审核报告", "审查", "审核", "批改报告", "合规检查", "挑错", "找错", "把关")),
    ("Tool_Recommend", ("推荐工具", "用什么工具", "管理工具推荐", "适用工具")),
]

# 疑问句/咨询句一律归 KB_QA（对齐 7028 classifier 的规则0：咨询句优先）
_QUESTION_OPENERS = ("什么是", "啥是", "是什么", "怎么", "如何", "有哪些",
                     "有哪几种", "介绍一下", "请问", "我想知道", "请告诉我")

# ── 各意图的必需槽位 ──────────────────────────────────────────────────
REQUIRED_SLOTS: dict[str, list[str]] = {
    "Generate_Report": ["report_type"],
    "Review_Report": ["file_ids"],
    "Optimize_Report": ["file_ids"],
    "Task_Status": [],
    "Tool_Recommend": [],
    "KB_QA": [],
}

# ── 追问文案 ──────────────────────────────────────────────────────────
SLOT_QUESTIONS: dict[str, str] = {
    "report_type": "请问报告类型是哪种？①创新型 ②问题解决型-指令性目标 ③问题解决型-自定目标",
    "file_ids": "请提供文件附件 ID（或先上传文件），我需要它才能继续",
    "session_id": "请提供任务号（session_id / job_id / task_id）",
}

_REPORT_TYPE_KWS: list[tuple[str, str]] = [
    ("创新型", "创新型"),
    ("指令性", "问题解决型-指令性目标"),
    ("自定目标", "问题解决型-自定目标"),
]

_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_LONG_ID_RE = re.compile(r"\b[0-9a-fA-F]{16,}\b")


def detect_intent(text: str) -> str:
    t = (text or "").strip()
    # 咨询句优先归 KB_QA：即使含"审查/生成"等动词，只要是在问概念就不该触发动作
    if any(t.startswith(q) for q in _QUESTION_OPENERS):
        return "KB_QA"
    for intent, kws in INTENT_RULES:
        if any(k in t for k in kws):
            return intent
    return "KB_QA"


def extract_slots(text: str, prev: Slots | None = None) -> Slots:
    """从用户本轮文本里抽取槽位，与已有槽位合并（已有值优先，不覆盖）。

    P1 只做确定性抽取：报告类型关键词 + 形如 UUID/长十六进制的附件 ID。
    更细的抽取（文件名、单位名）P2 交给 LLM。
    """
    slots: Slots = dict(prev or {})
    t = text or ""

    if "report_type" not in slots or not slots.get("report_type"):
        for kw, value in _REPORT_TYPE_KWS:
            if kw in t:
                slots["report_type"] = value
                break

    if not slots.get("file_ids"):
        ids = _UUID_RE.findall(t) + _LONG_ID_RE.findall(t)
        if ids:
            slots["file_ids"] = ids

    return slots


def _has_task_pointer(slots: Slots) -> bool:
    return any(slots.get(k) for k in ("session_id", "job_id", "task_id"))


def node(state: QCAgentState, config=None) -> dict:
    """路由节点：产出 intent / slots / missing。"""
    user_text = ""
    for m in reversed(state.get("messages", [])):
        # 真实链路里是 BaseMessage；单测时可能直接传 dict，两种都要能取到
        if isinstance(m, dict):
            role, content = m.get("role", ""), m.get("content", "")
        else:
            role, content = getattr(m, "type", "") or "", getattr(m, "content", "") or ""
        if role in ("human", "user"):
            user_text = content
            break

    intent = detect_intent(user_text)
    slots = extract_slots(user_text, state.get("slots"))

    missing: list[str] = []
    if intent == "Task_Status" and not _has_task_pointer(slots):
        missing.append("session_id")
    else:
        for slot in REQUIRED_SLOTS.get(intent, []):
            if not slots.get(slot):
                missing.append(slot)

    return {"intent": intent, "slots": slots, "missing": missing}


def question_for(missing: list[str]) -> str:
    qs = [SLOT_QUESTIONS.get(m, f"请补充：{m}") for m in missing]
    return "；".join(qs)


def slots_brief(slots: Slots | None) -> str:
    return json.dumps(slots or {}, ensure_ascii=False)
