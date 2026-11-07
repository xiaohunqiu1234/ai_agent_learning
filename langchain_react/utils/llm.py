"""LLM 接入 —— 复用 ReAct_local.py 最有价值的一条设计：**实测探测**再决定工具调用路线。

同一个 7888 代理上，不同模型对 OpenAI tools 协议的支持情况不一样，
按模型名字猜迟早猜错。所以：
    supports_tools() == True  -> bind_tools，参数由服务端按 schema 产出（稳）
    supports_tools() == False -> 文本 ReAct，靠提示词约束 Thought/Action 格式（兼容性最好）

两条路线的提示词与解析都在 prompts/react.py，图里无缝切换。
"""

from __future__ import annotations

import os
import sys

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from qc_agent_mcp.configuration import Configuration

# ── 工具能力探测结果缓存（同一进程只探测一次）────────────────────────
_TOOLS_CACHE: dict[tuple[str, str], bool] = {}


class ProxyChatModel(BaseChatModel):
    """直连 7888 /v1/chat/completions 的最小实现。

    存在的意义：当环境装不了 langchain-openai（或代理协议有偏差）时，
    仍能跑通"纯文本"链路，不至于整条图起不来。
    """

    # pydantic v2 默认把 "model_" 视为保护命名空间，这里显式放开
    model_config = {"protected_namespaces": ()}

    base_url: str = "http://localhost:7888/v1"
    api_key: str = ""
    model_name: str = "default"
    temperature: float = 0.2
    timeout: float = 120.0

    @property
    def _llm_type(self) -> str:
        return "qc_proxy_chat"

    def _default_params(self) -> dict:
        return {}

    @staticmethod
    def _to_payload_messages(messages) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            t = getattr(m, "type", "human")
            role = "system" if t == "system" else ("assistant" if t in ("ai", "assistant") else "user")
            out.append({"role": role, "content": getattr(m, "content", "") or ""})
        return out

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        payload = {
            "model": self.model_name,
            "messages": self._to_payload_messages(messages),
            "stream": False,
            "temperature": self.temperature,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout, verify=False) as client:
            resp = client.post(self.chat_completions_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def chat_completions_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


def llm_base_url() -> str:
    return os.getenv("QC_LLM_BASE", "http://localhost:7888/v1").rstrip("/")


def build_llm(cfg: Configuration | None = None) -> BaseChatModel:
    """构造聊天模型。

    provider:
        auto（默认）  —— 优先 langchain-openai（支持 bind_tools），装不上则退回 ProxyChatModel
        openai_compat —— 强制 langchain-openai
        proxy         —— 强制内置 httpx 客户端
    """
    cfg = cfg or Configuration()
    base = llm_base_url()
    key = os.getenv("QC_LLM_API_KEY", "")
    provider = os.getenv("QC_LLM_PROVIDER", "auto").lower()

    if provider in ("auto", "openai_compat"):
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                base_url=base,
                api_key=key or "EMPTY",
                model=cfg.model,
                temperature=cfg.temperature,
                timeout=cfg.llm_timeout,
                max_retries=1,
            )
        except ImportError:
            if provider == "openai_compat":
                raise
            # MCP stdio 模式下 stdout 归协议所有，提示必须走 stderr
            print(
                "[提示] langchain-openai 不可用，退回内置 httpx 客户端（文本 ReAct 路线）",
                file=sys.stderr,
            )

    return ProxyChatModel(
        base_url=base,
        api_key=key,
        model_name=cfg.model,
        temperature=cfg.temperature,
        timeout=cfg.llm_timeout,
    )


def _probe_tools_support(base: str, key: str, model: str) -> bool:
    """实测：发一个带 tools 的最小请求，看服务端是否拒绝。

    只看是否 4xx/5xx——相当多的兼容网关会悄悄忽略不认识的参数而不报错，
    因此这里再补一道：返回体里出现 tool_calls 字段才算真支持。
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "__probe__",
                    "description": "probe",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "max_tokens": 16,
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        with httpx.Client(timeout=20, verify=False) as client:
            resp = client.post(base.rstrip("/") + "/chat/completions", json=payload, headers=headers)
        if resp.status_code >= 400:
            return False
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {}) or {}
        return ("tool_calls" in msg) or bool(msg.get("tool_calls"))
    except Exception:
        return False


def supports_tools(llm: BaseChatModel | None = None, force: bool = False) -> bool:
    """模型是否支持 OpenAI tools 协议。

    QC_LLM_TOOLS=on / off 可强制指定；auto 时做一次真实探测并缓存。
    """
    override = os.getenv("QC_LLM_TOOLS", "auto").lower()
    if override in ("1", "true", "on", "yes"):
        return True
    if override in ("0", "false", "off", "no"):
        return False

    base = llm_base_url()
    model = os.getenv("QC_LLM_MODEL", "default")
    if llm is not None:
        model = getattr(llm, "model_name", None) or getattr(llm, "model", None) or model

    cache_key = (base, model)
    if not force and cache_key in _TOOLS_CACHE:
        return _TOOLS_CACHE[cache_key]

    result = _probe_tools_support(base, os.getenv("QC_LLM_API_KEY", ""), model)
    _TOOLS_CACHE[cache_key] = result
    return result
