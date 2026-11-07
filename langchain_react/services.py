"""服务地址表与端口（Agent 层专用）。

为什么不复用旧 config/config.py：
    旧模块在**导入时**就执行 LOG_DIR.mkdir()（且写死 /home/QC_AGENT/logs），
    在非部署机（如开发机 macOS）上会直接抛权限异常。Agent 层是新进程，
    不该被这种导入副作用拖累，因此这里独立维护一份地址表。

代价是环境地址要维护两处，但只有 4 行，且可用 QC_ENV_URL 直接覆盖。
"""

from __future__ import annotations

import os

# ── 环境基址（与 config/config.py 保持一致）──────────────────────────
ENV_URLS: dict[str, str] = {
    "local": "http://localhost",
    "dev": "http://172.16.56.49",
    "sim": "http://172.16.144.28",
    "prod": "http://10.9.73.155",
}

DEFAULT_ENV = os.getenv("QC_ENV", "local")


def env_base(env: str | None = None) -> str:
    """取当前环境基址；QC_ENV_URL 可强制覆盖（容器注入场景）。"""
    override = os.getenv("QC_ENV_URL", "").strip()
    if override:
        return override.rstrip("/")
    return ENV_URLS.get(env or DEFAULT_ENV, ENV_URLS["local"])


# ── 服务表：名称 -> (端口, 健康检查路径) ──────────────────────────────
# 健康检查路径取自各服务现有路由，新增服务时在此登记即可。
SERVICE_TABLE: dict[str, tuple[int, str]] = {
    "llm_proxy": (7888, "/health"),  # 7888 的健康路径需按实际实现校核
    "tool_recommend": (7001, "/qc/ai/tool_recommend/health"),
    "report_generation": (7022, "/qc/ai/report_generation/health"),
    "qc_review": (7025, "/qc/ai/qc_review/health"),
    "general_qa": (7026, "/qc/ai/general_chat/health"),
    "ai_review": (7027, "/qc/ai/evaluation/health"),
    "intent_judgment": (7028, "/qc/ai/intent/health"),
    "report_optimization": (7029, "/qc/ai/health"),
    "file_middleware": (7031, "/health"),  # 7031 的健康路径需按实际实现校核
}


def service_url(name: str, path: str = "", env: str | None = None) -> str:
    """拼完整 URL：http://host:port + path"""
    if name not in SERVICE_TABLE:
        raise KeyError(f"未登记的服务: {name}（可用: {sorted(SERVICE_TABLE)}）")
    port, _ = SERVICE_TABLE[name]
    return f"{env_base(env)}:{port}{path}"


def health_url(name: str, env: str | None = None) -> str:
    """取某服务的健康检查 URL。"""
    port, health_path = SERVICE_TABLE[name]
    return f"{env_base(env)}:{port}{health_path}"


# ── Agent 自身端口 ────────────────────────────────────────────────────
AGENT_API_PORT = int(os.getenv("QC_AGENT_PORT", "7040"))
MCP_HTTP_PORT = int(os.getenv("QC_MCP_PORT", "7041"))
MCP_TRANSPORT = os.getenv("QC_MCP_TRANSPORT", "stdio")
