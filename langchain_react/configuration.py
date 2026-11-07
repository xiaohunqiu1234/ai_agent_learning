"""运行时可调参数 —— LangGraph 标准做法：由 config["configurable"] 注入。

节点一律通过 Configuration.from_runnable_config(config) 取值，
这样同一张图可以在不同 thread 上用不同参数跑（调试时尤其有用）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any


def _get_env(name: str, default: Any, cast=str) -> Any:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


@dataclass(kw_only=True)
class Configuration:
    """智能体可调参数。优先取 config["configurable"]，回落到环境变量。"""

    # ── LLM ──────────────────────────────────────────────
    model: str = _get_env("QC_LLM_MODEL", "default")
    temperature: float = _get_env("QC_LLM_TEMPERATURE", 0.2, float)
    llm_timeout: float = _get_env("QC_LLM_TIMEOUT", 120.0, float)

    # ── ReAct 循环护栏（替代 AgentExecutor 的 max_iterations）──
    max_iterations: int = _get_env("QC_AGENT_MAX_ITERATIONS", 5, int)

    # ── 工具 Observation 截断，防止打爆上下文 ────────────────
    observation_max_chars: int = _get_env("QC_AGENT_OBSERVATION_MAX_CHARS", 3000, int)

    # ── 调用方身份（用于长期记忆 namespace）──────────────────
    user_id: str = ""

    @classmethod
    def from_runnable_config(cls, config: dict | None = None) -> "Configuration":
        """从 LangGraph 的 RunnableConfig 里抽取本类声明的字段。"""
        configurable = (config or {}).get("configurable", {}) or {}
        values: dict[str, Any] = {}
        for f in fields(cls):
            if f.name in configurable and configurable[f.name] is not None:
                values[f.name] = configurable[f.name]
        return cls(**values)
