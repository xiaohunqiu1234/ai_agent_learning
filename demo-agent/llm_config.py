"""LLM 配置加载器。

约定：
  - `.env` 与本章脚本放在一起（都在 `02-demo-agent/` 下），配置固定按本文件所在目录加载，
    因此无论从哪个工作目录运行脚本，拿到的都是这一份配置。
  - 业务脚本只需要一行 `from llm_config import build_llm`，切换模型只改 `.env` 里的 MODEL_PROVIDER。

自检命令（在 02-demo-agent 目录执行）：
    python llm_config.py                    # 打印当前配置 + 服务连通性
    MODEL_PROVIDER=openai python llm_config.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

# 配置固定按本文件所在目录加载（.env 与脚本同目录），
# 这样即使从别的工作目录运行脚本，也一定拿到同一份配置。
CONFIG_DIR = Path(__file__).resolve().parent
load_dotenv(CONFIG_DIR / ".env", override=False)

# 全部支持的提供方
PROVIDERS = (
    "openai",
    "deepseek",
    "ollama",
    "dashscope",
    "ollama_direct",
    "qwen_vl",
    "gateway",
    "ollama_native",
    "huggingface",
)

# 旧名兼容：曾指向 7888 网关的 local / local_openai 已废弃，统一落到直连 Ollama，
# 免得老 .env 一升级就报"未知的 MODEL_PROVIDER"。
_PROVIDER_ALIAS = {"local": "ollama_direct", "local_openai": "ollama_direct"}


def get_provider(provider: str | None = None) -> str:
    """取得当前提供方名称，优先用参数，其次读 MODEL_PROVIDER。"""
    raw = (provider or os.getenv("MODEL_PROVIDER") or os.getenv("LLM_PROVIDER") or "ollama_direct")
    name = raw.strip().lower()
    name = _PROVIDER_ALIAS.get(name, name)
    if name not in PROVIDERS:
        raise ValueError(f"未知的 MODEL_PROVIDER: '{raw}'。支持的值: {', '.join(PROVIDERS)}")
    return name


def _provider_settings(provider: str) -> dict[str, Any]:
    """按提供方解析出 base_url / api_key / model。"""
    common: dict[str, Any] = {
        "temperature": float(os.getenv("MODEL_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0"))),
        "max_tokens": int(os.getenv("MODEL_MAX_TOKENS", os.getenv("LLM_MAX_TOKENS", "2000"))),
        "top_p": float(os.getenv("MODEL_TOP_P", "1.0")),
        "timeout": float(os.getenv("MODEL_TIMEOUT", os.getenv("LLM_TIMEOUT", "180"))),
        "max_retries": int(os.getenv("MODEL_MAX_RETRIES", "5")),
    }

    if provider == "openai":
        return {
            **common,
            "base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        }
    if provider == "deepseek":
        return {
            **common,
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        }
    if provider == "dashscope":
        return {
            **common,
            "base_url": os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            "api_key": os.getenv("DASHSCOPE_API_KEY", ""),
            "model": os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
        }
    if provider == "ollama":  # Ollama 的 OpenAI 兼容端口
        return {
            **common,
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            "api_key": os.getenv("OLLAMA_API_KEY", "not-needed"),
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        }
    if provider == "gateway":  # 内网 OpenAI 兼容网关：只代理 /v1/chat/completions
        return {
            **common,
            "base_url": os.getenv("GATEWAY_BASE_URL", "http://192.168.3.10:7889/v1"),
            "api_key": os.getenv("GATEWAY_API_KEY", "not-needed"),
            # 网关通常忽略请求里的 model 字段，真实模型由服务端决定：
            # 这里填的只是"请求里署的名"，不影响实际调用的是谁
            "model": os.getenv("GATEWAY_MODEL", "qwen2.5-vl"),
        }
    if provider in ("ollama_direct", "qwen_vl"):  # 直连 Ollama，不经过网关
        # 两者指向同一个服务，区别只在默认模型：
        #   ollama_direct -> deepseek-r1:32b：推理链完整，但 Ollama 明确不支持 tools
        #   qwen_vl       -> qwen3-vl:8b ：支持 OpenAI tools 协议，也能看图
        model_env = "OLLAMA_DIRECT_MODEL" if provider == "ollama_direct" else "QWEN_VL_MODEL"
        default_model = "deepseek-r1:32b" if provider == "ollama_direct" else "qwen3-vl:8b"
        return {
            **common,
            "base_url": os.getenv("OLLAMA_DIRECT_BASE_URL", "http://192.168.3.10:11434/v1"),
            "api_key": os.getenv("OLLAMA_DIRECT_API_KEY", "not-needed"),
            "model": os.getenv(model_env, default_model),
        }
    if provider == "ollama_native":
        return {
            **common,
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").replace("/v1", ""),
            "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"),
        }
    if provider == "huggingface":
        return {
            **common,
            "model": os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        }
    raise ValueError(f"未处理的提供方: {provider}")


def _parse_extra_body(raw: str | None) -> dict[str, Any]:
    """解析额外请求体字段，支持 JSON 或 `k=v,k2=v2` 简化写法。

    某些网关需要非标准字段（例如本项目的 LLM 代理服务要求 output_mode=response），
    这些字段必须走 LangChain 的 extra_body 下发。
    """
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        import json

        return json.loads(text)
    out: dict[str, Any] = {}
    for item in text.split(","):
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        low = value.lower()
        if low in ("true", "false"):
            out[key] = low == "true"
        elif low.replace(".", "", 1).isdigit():
            out[key] = float(value) if "." in value else int(value)
        else:
            out[key] = value
    return out


def build_http_client(timeout: float, trust_env: bool) -> httpx.Client:
    """构造 httpx 客户端。

    MODEL_TRUST_ENV=0 时 trust_env=False，忽略 shell 里的 http_proxy/https_proxy，
    避免本地代理没启动就报 Connection refused；内网地址另靠 .env 里的 no_proxy 兜底。
    """
    return httpx.Client(timeout=timeout, trust_env=trust_env, follow_redirects=True)


def build_llm(provider: str | None = None, **overrides: Any):
    """按 MODEL_PROVIDER 构造 LangChain 聊天模型（统一返回 BaseChatModel 接口用法）。"""
    name = get_provider(provider)
    cfg = _provider_settings(name)
    cfg.update(overrides)

    # openai / deepseek / dashscope / ollama / ollama_direct / qwen_vl / gateway 都讲 OpenAI 协议
    if name in ("openai", "deepseek", "dashscope", "ollama", "ollama_direct", "qwen_vl", "gateway"):
        from langchain_openai import ChatOpenAI

        trust_env = os.getenv("MODEL_TRUST_ENV", "0").strip().lower() not in ("0", "false", "no")
        kwargs: dict[str, Any] = {
            "model": cfg["model"],
            "base_url": cfg["base_url"],
            "api_key": cfg["api_key"],
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "top_p": cfg["top_p"],
            "timeout": cfg["timeout"],
            "max_retries": cfg["max_retries"],
            "http_client": build_http_client(cfg["timeout"], trust_env),
        }
        # 个别服务只接受 stream=true；自定义字段走 extra_body
        if cfg.get("streaming"):
            kwargs["streaming"] = True
        if cfg.get("extra_body"):
            kwargs["extra_body"] = cfg["extra_body"]
        return ChatOpenAI(**kwargs)

    if name == "ollama_native":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "使用 MODEL_PROVIDER=ollama_native 前请先安装：pip install -U langchain-ollama"
            ) from e
        return ChatOllama(
            model=cfg["model"],
            base_url=cfg["base_url"],
            temperature=cfg["temperature"],
            num_predict=cfg["max_tokens"],
        )

    if name == "huggingface":
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        except ImportError as e:
            raise ImportError(
                "使用 MODEL_PROVIDER=huggingface 前请先安装："
                "pip install -U langchain-huggingface transformers torch accelerate"
            ) from e
        pipe = HuggingFacePipeline.from_model_id(
            model_id=cfg["model"],
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": cfg["max_tokens"],
                "do_sample": cfg["temperature"] > 0,
                "temperature": cfg["temperature"] or None,
            },
        )
        return ChatHuggingFace(llm=pipe)

    raise ValueError(f"未处理的提供方: {name}")


# 与旧代码保持兼容的别名
get_llm = build_llm


# ============================================================
# 能力探测：当前模型支不支持 OpenAI tools（function calling）协议
# ============================================================

# 探针工具，名字刻意做成业务里绝不会出现的样子
_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "__capability_probe__",
        "description": "仅用于探测服务端是否支持 tools 协议",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_TOOL_SUPPORT_CACHE: dict[tuple[str, str], bool] = {}


def _llm_endpoint(llm) -> tuple[str, str, str]:
    """从已构造好的模型对象里取出 endpoint / model / key。"""
    if llm is None:
        llm = build_llm()
    base = str(getattr(llm, "openai_api_base", "") or "").rstrip("/")
    model = str(getattr(llm, "model_name", "") or "")
    key = getattr(llm, "openai_api_key", "")
    if hasattr(key, "get_secret_value"):  # pydantic SecretStr
        key = key.get_secret_value()
    return base, model, key or "not-needed"


def supports_tools(llm=None, *, refresh: bool = False) -> bool:
    """探测当前模型是否支持 OpenAI tools（function calling）协议。

    判定依据是**服务端有没有拒绝**，而不是这一次有没有真的发起调用：
      - Ollama 遇到不支持的模型会直接 400 + "does not support tools"（如 deepseek-r1:32b）
      - 支持的模型正常返回 200；但"这次有没有触发调用"不能拿来当依据——
        它取决于 token 预算和提问内容，连显式 tool_choice 在 Ollama 上都不一定生效。

    结果按 (base_url, model) 缓存。想连这一次探测请求都省掉，
    可在 .env 里写 MODEL_SUPPORTS_TOOLS=0 / 1 直接指定。
    """
    override = (os.getenv("MODEL_SUPPORTS_TOOLS", "") or "").strip().lower()
    if override in ("0", "1", "true", "false", "yes", "no"):
        return override in ("1", "true", "yes")

    base, model, key = _llm_endpoint(llm)
    if not base or not model:
        # 拿不到 OpenAI 端点，说明不是这一族协议（例如 transformers 本机推理）
        return False

    cache_key = (base, model)
    if not refresh and cache_key in _TOOL_SUPPORT_CACHE:
        return _TOOL_SUPPORT_CACHE[cache_key]

    trust_env = os.getenv("MODEL_TRUST_ENV", "0").strip().lower() not in ("0", "false", "no")
    result = False
    try:
        with build_http_client(30, trust_env) as client:
            resp = client.post(
                base + "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "tools": [_PROBE_TOOL],
                    "tool_choice": "auto",
                    "max_tokens": 16,
                },
            )
            result = resp.status_code < 400
    except Exception:
        result = False

    _TOOL_SUPPORT_CACHE[cache_key] = result
    return result


def check_server(base_url: str | None = None) -> tuple[bool, str]:
    """探测 OpenAI 兼容服务是否在线。

    分级判断，因为我们会遇到三类后端：
      1. 完整实现 OpenAI 协议（vLLM / Ollama / LM Studio）：GET /models 返回 200 与模型列表
      2. 只代理 /v1/chat/completions 的内网网关（如本项目的 LLM 代理服务）：
         /models 返回 404，此时退一步用 /health 或根路径判断是否可达
      3. 服务压根没起：连接被拒 / 超时
    """
    base = (base_url or _provider_settings(get_provider()).get("base_url") or "").rstrip("/")
    if not base:
        return False, "未配置 base_url"

    # http://host:port/v1 -> http://host:port（网关的健康检查接口通常在根路径）
    import re

    root = re.sub(r"/v\d+$", "", base)

    trust_env = os.getenv("MODEL_TRUST_ENV", "0").strip().lower() not in ("0", "false", "no")
    timeout = float(os.getenv("MODEL_TIMEOUT", "180"))
    try:
        with build_http_client(timeout, trust_env) as client:
            resp = client.get(base + "/models")
            if resp.status_code == 200:
                try:
                    models = [m.get("id", "?") for m in resp.json().get("data", [])]
                except Exception:
                    models = []
                return True, f"在线，可用模型 {len(models)} 个：" + ", ".join(models[:10])

            # /models 不可用，退一步：只要健康检查或根路径有应答就认为在线
            for probe in (root + "/health", root + "/"):
                try:
                    r = client.get(probe)
                except Exception:
                    continue
                if r.status_code < 500:
                    return True, (
                        f"在线（该网关不提供 /models，{probe} 返回 {r.status_code}）；"
                        f"模型名以服务端配置为准，请求里的 model 字段不影响实际使用"
                    )
            return False, f"服务无响应：/models -> {resp.status_code} {resp.text[:200]}"
    except Exception as e:  # 连接被拒 / 超时
        hint = ""
        if trust_env is False and os.getenv("https_proxy"):
            hint = "（当前 MODEL_TRUST_ENV=0 已绕过代理；若服务需走代理请改为 1）"
        return False, f"无法连接 {base}（{type(e).__name__}：{str(e)[:120]}）{hint}"


def describe() -> str:
    """当前配置的一行摘要。"""
    name = get_provider()
    cfg = _provider_settings(name)
    return (
        f"provider={name} model={cfg.get('model')} target={cfg.get('base_url', '本机 transformers')} "
        f"temp={cfg.get('temperature')} max_tokens={cfg.get('max_tokens')} timeout={cfg.get('timeout')}s"
    )


if __name__ == "__main__":
    print("配置文件：", CONFIG_DIR / ".env")
    print("当前配置：", describe())
    ok, msg = check_server()
    print("服务探测：", "✅" if ok else "❌", msg)
    if not ok:
        print("提示：换个提供方试试，例如 MODEL_PROVIDER=openai python llm_config.py")
