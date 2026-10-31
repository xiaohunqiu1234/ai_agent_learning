"""多轮聊天示例：模型来源交给同目录下的 `llm_config.py`。

端点 / Key / 模型名全部来自本目录的 .env，改 MODEL_PROVIDER 即可切换；
本文件自己不读任何环境变量，也不为某个提供方写分支。

每一轮发出的 `messages` 由「人设 + 历史 + 本轮问题」三部分组成。
语言模型是无状态的，所谓"记得上一轮"就是客户端每轮把整段历史重发一遍——
LangChain 的 PromptTemplate、Memory 组件本质上都是在自动拼这个数组。

运行（都在这一个目录里）：
    python llm_config.py          # 自检：打印当前配置 + 服务是否在线
    python Chat_local.py

进入后可以连续提问，上下文会带进下一轮。命令：
    exit / quit   退出
    clear         清空上下文（回到开场那段示例历史）
    json          切换结构化输出：开启时要求模型只输出 JSON
"""

from __future__ import annotations

import json
import re

# llm_config.py / .env 都在本目录下，直接按模块名导入即可（cwd 不影响，配置按其所在目录加载）
from llm_config import build_llm, describe, get_provider

# ---------------------------------------------------------------------------
# 每一轮要发的 messages = 人设 + 历史 + 本轮问题
#   system                    -> 设定人设与输出要求
#   历史（user/assistant 交替） -> 之前各轮累积下来的上下文
#   user                      -> 当前这一轮的问题
# ---------------------------------------------------------------------------
SYSTEM = "您是一个帮助用户了解鲜花信息的智能助手,并能够输出JSON格式的内容。"
# 关掉结构化输出时换这句，否则人设里仍写着"输出 JSON"，切换就不生效了
SYSTEM_PLAIN = "您是一个帮助用户了解鲜花信息的智能助手。"

# 开场先垫一段写好的历史，让第一轮提问就已经处在多轮语境里。
# 这一问一答正是"历史上下文"在 messages 里的样子。
INITIAL_HISTORY: list[dict[str, str]] = [
    {"role": "user", "content": "生日送什么花最好？"},
    {"role": "assistant", "content": "玫瑰是生日礼物的热门选择。"},
]

# 部分 OpenAI 兼容服务不实现 response_format 参数，降级时用它约束输出
JSON_FALLBACK_HINT = (
    "你必须且只能输出一个 JSON 对象，不要输出任何解释文字、代码块标记或换行外的多余内容。"
    '键名用英文，例如：{"answer": "答复内容", "reason": "理由", "delivery_time": "所需时长"}'
)


def build_llm_with_local_config():
    """按 .env 构造聊天模型。

    build_llm 对 openai / deepseek / dashscope / ollama / ollama_direct /
    qwen_vl / gateway 都返回同一个 LangChain BaseChatModel 接口，
    因此本文件不用写任何分支。
    """
    provider = get_provider()
    print(f"当前模型来源：{provider}")
    print(f"连接参数：{describe()}")
    return build_llm(provider)


def call_once(llm, messages: list[dict[str, str]], *, json_mode: bool):
    """发起一次请求。json_mode=True 时附加协议级 response_format 参数。"""
    target = llm.bind(response_format={"type": "json_object"}) if json_mode else llm
    try:
        return target.invoke(messages)
    except Exception as exc:  # noqa: BLE001 - 各服务对未知参数的报错类型不一致
        # 少数服务会因不认识 response_format 直接报 400，这里退回到普通调用
        if not json_mode:
            raise
        print(
            f"\n[降级] 当前服务拒绝 response_format=json_object（{type(exc).__name__}），改用普通参数重试。"
        )
        return llm.invoke(messages)


def extract_json(content: str) -> dict | None:
    """把模型输出里的 JSON 抠出来。

    本地小模型的常见毛病：外面套一层 ```json 代码块，前后还带寒暄正文。
    所以先剥代码块，再截取第一个花括号包裹的片段。
    """
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if not text.startswith("{"):
        block = re.search(r"\{.*\}", text, re.DOTALL)
        if not block:
            return None
        text = block.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def build_messages(system: str, history: list[dict[str, str]], question: str):
    """把「人设 + 历史 + 本轮问题」拼成一次请求要发的 messages。

    多轮的本质就在这三行：模型不记得上一轮说过什么，
    历史必须由客户端每轮重新拼装、整段重发。
    """
    return [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": question},
    ]


def chat(llm, messages: list[dict[str, str]]):
    """拿到一段可解析成 JSON 的回答，必要时自动重试一次。

    两级策略，顺序不能反：
      1. 协议级：response_format=json_object（OpenAI 协议原生支持，最稳）
      2. 提示词级：把 JSON 要求写进 system——部分兼容服务只吃这一套

    判断依据是"结果能不能解析"，而不是"有没有报错"：
    相当多的兼容服务会悄悄忽略 response_format（不报错也不生效），
    只看异常的话会误以为第 1 级成功了。
    """
    response = call_once(llm, messages, json_mode=True)
    if extract_json(getattr(response, "content", "")) is not None:
        return response, "response_format"

    print("\n[降级] 返回内容不是合法 JSON（服务多半忽略了 response_format），改用提示词约束重试。")
    patched = [dict(m) for m in messages]
    patched[0]["content"] = f"{SYSTEM} {JSON_FALLBACK_HINT}"
    return call_once(llm, patched, json_mode=False), "提示词约束"


def render_usage(response) -> str:
    """取服务端返回的用量；部分服务不返回 usage，此时给出空串。"""
    meta = getattr(response, "response_metadata", {}) or {}
    usage = meta.get("usage") or meta.get("token_usage")
    if not usage:
        return ""
    total = usage.get("total_tokens")
    return f"本轮 tokens：{total}" if total else f"用量：{usage}"


HELP = "已进入多轮对话。exit 退出，clear 清空上下文，json 切换结构化输出。"


def main() -> None:
    llm = build_llm_with_local_config()

    history: list[dict[str, str]] = [dict(m) for m in INITIAL_HISTORY]
    structured = True  # 是否要求模型只输出 JSON，可用 json 命令切换
    print("\n" + HELP)

    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出")
            break

        if not question:
            continue
        lowered = question.lower()

        if lowered in ("exit", "quit", "退出"):
            print("已退出")
            break
        if lowered in ("clear", "清空"):
            history = [dict(m) for m in INITIAL_HISTORY]
            print("上下文已清空（回到开场状态）")
            continue
        if lowered == "json":
            structured = not structured
            print(f"结构化输出：{'开' if structured else '关'}")
            continue

        # 人设也要跟着开关换，否则关掉结构化后人设里仍写着"输出 JSON"
        messages = build_messages(
            SYSTEM if structured else SYSTEM_PLAIN, history, question
        )
        if structured:
            response, mode = chat(llm, messages)
            print(f"[JSON 约束方式：{mode}]")
        else:
            response = call_once(llm, messages, json_mode=False)

        content = getattr(response, "content", "")
        print(f"助手：{content}")

        if structured:
            data = extract_json(content)
            if data is not None:
                print("解析结果：")
                print(json.dumps(data, ensure_ascii=False, indent=2))
            else:
                print("解析结果：仍不是合法 JSON——本地小模型的常见现象，"
                      "可把 temperature 调低或换更大模型")

        # 只有问答进历史，上面那些调试信息不进，免得污染上下文
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": content})

        usage_text = render_usage(response)
        if usage_text:
            print(usage_text)
        print(f"上下文：{len(history)} 条（每轮整段重发）")


if __name__ == "__main__":
    main()
