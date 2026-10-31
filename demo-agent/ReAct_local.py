"""用本地模型跑 ReAct Agent 的示例（支持多轮上下文问答）。

模型来源完全由本目录下的 `.env` 决定：
    MODEL_PROVIDER=ollama_direct  -> 直连 Ollama，默认 deepseek-r1:32b（当前默认）
    MODEL_PROVIDER=qwen_vl        -> 直连同一个 Ollama，默认 qwen3-vl:8b
    MODEL_PROVIDER=openai         -> 官方云 API
先在本目录执行 `python llm_config.py` 确认服务在线，再跑本脚本。

进入后可以连续提问，上下文会带进下一轮；输入 exit 退出，clear 清空上下文。

能否做"工具调用"要看模型自己支不支持 OpenAI tools 协议，本脚本会自动二选一：
    supports_tools() == True   -> create_tool_calling_agent（结构化调用，服务端校验参数）
    supports_tools() == False  -> 文本解析版 ReAct（靠 Question/Thought/Action 格式约束）
同一个 Ollama 上 deepseek-r1:32b 属后者、qwen3-vl:8b 属前者，别按模型名字猜。

本地化 / 多轮的两个注意点已在本文件中处理：
  - 提示词默认读本地副本 `prompts/react_chat.py`（官方 chat 版的逐字拷贝），
    因此**断网也能用上官方提示词**；本地副本缺失才联网 hub.pull，都失败才回落中文版
  - 打开 handle_parsing_errors，让模型偶发格式错误时能自我纠正而不是直接崩

另注：这两个模型都是推理模型，token 会先花在 reasoning 上，
      .env 的 MODEL_MAX_TOKENS 若给太小（如原来的 2000），容易在拿到正文前就被截断。
"""

from __future__ import annotations

from datetime import datetime

# llm_config.py / react_compat.py / .env 都在本目录下，直接导入即可
from langchain_classic import hub  # noqa: E402
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent  # noqa: E402
from langchain_community.tools import DuckDuckGoSearchRun  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate  # noqa: E402
from langchain_core.tools import StructuredTool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from llm_config import build_llm, check_server, describe, get_provider, supports_tools  # noqa: E402
from react_compat import build_react_agent  # noqa: E402

# --- 工具入参 schema -------------------------------------------------------
# 为什么不用 langchain_core.tools.Tool：
#   它的 func 模式只接受"单个入参"，一旦模型给出空字典（如无参工具的组织形式
#   是 arguments = "{}"），会在调用前就抛 ToolException: Too many arguments。
#   改签名绕不过去——报错发生在绑定层，不在函数体里。
# StructuredTool 允许入参是 dict / 空串 / 缺字段，两种形态都能吃下。


class SearchArgs(BaseModel):
    """Search 的入参。给默认值，模型漏填时不至于直接失败。"""

    query: str = Field(default="", description="要搜索的关键词或问题")


class NoArgs(BaseModel):
    """无入参工具的空 schema：显式声明后，空字典入参才能被接受。"""


# 最后兜底用的中文版 ReAct 提示词（本地副本与 hub 都拿不到时才登场）。
# 它没有 chat_history 占位符，落到它身上时多轮记忆会失效——
# 所以提示词来源的第一顺位是 prompts/react_chat.py。
REACT_PROMPT_CN = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template=(
        "尽你所能回答下面的问题，你可以使用以下工具：\n\n{tools}\n\n"
        "请严格使用下面的格式回答：\n\n"
        "Question: 你必须回答的输入问题\n"
        "Thought: 你应该始终思考下一步该做什么\n"
        "Action: 要执行的动作，必须是 [{tool_names}] 中的一个\n"
        "Action Input: 传给该动作的输入参数；不需要参数就写 None\n"
        "Observation: 动作执行的结果\n"
        "...（Thought/Action/Action Input/Observation 可以重复 N 次）\n"
        "Thought: 我现在知道最终答案了\n"
        "Final Answer: 对原始输入问题的最终答案\n\n"
        "现在开始！\n\n"
        "Question: {input}\n"
        "Thought:{agent_scratchpad}"
    ),
)

# 走 OpenAI tools 协议时用的提示词。
# 文本解析版 ReAct 不需要它——那条路线靠 Question/Thought/Action 的文字格式约束模型。
TOOL_CALLING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是花语秘境的智能助手。有合适的工具就调用工具，并依据工具返回的结果作答；"
            "没有合适工具时直接回答。请用中文回复。",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# 联网搜索工具（DuckDuckGo，无需 Key）；纯内网环境可注释掉本行和下方的 Search 工具
search = DuckDuckGoSearchRun()


def safe_search(query: str = "") -> str:
    """包装 DuckDuckGo，避免搜索引擎超时时把整个 Agent 链路炸掉。

    返回错误文本而不是抛异常，Agent 就能把"搜索失败"当作一次 Observation 继续推理。
    """
    query = (query or "").strip()
    # 模型有时把"没有参数"写成字符串 "None"，直接拿去搜会得到无意义结果
    if query in ("", "None", "none", "null"):
        return "搜索失败：没有提供搜索关键词，请给出要查询的具体内容后重试。"
    try:
        return search.run(query)
    except Exception as e:  # ddgs 在受限网络下会抛 TimeoutException / RateLimitException
        hint = ""
        if "timed out" in str(e).lower() or "Timeout" in type(e).__name__:
            hint = "（多为网络受限所致：可配置代理后重试，或改用 Tavily 等搜索源）"
        return f"搜索失败：{type(e).__name__}: {str(e)[:200]} {hint}".strip()


def get_current_time(_: str = "") -> str:
    """离线也能验证工具调用是否生效的小工具。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_prompt():
    """ReAct 提示词来源：本地副本 → 联网拉取 → 中文兜底。

    本地副本（`prompts/react_chat.py`）排第一，是为了让"能不能跑起来"
    彻底不依赖网络：hub.pull 走 api.smith.langchain.com，内网/断网时会失败，
    而中文兜底版没有 `chat_history` 占位符，会让多轮记忆失效。

    注意：这里**不做** `.partial(chat_history=[])`——多轮问答需要每轮把真实历史
    传进 `chat_history` 占位符，固定写死就记不住上一轮说过什么了。
    """
    try:
        from prompts.react_chat import load_react_chat

        return load_react_chat()
    except Exception as e:
        print(f"[提示] 本地提示词副本不可用（{e}），尝试联网拉取官方版")
    try:
        return hub.pull("hwchase17/react-chat")
    except Exception as e:
        print(f"[提示] hub 拉取失败，回落到本地中文 prompt：{e}")
        print("       该兜底版没有 chat_history 占位符，多轮记忆不生效；")
        print("       保留 prompts/react_chat.py 即可恢复多轮。")
        return REACT_PROMPT_CN


def build_agent(llm, tools, react_prompt, *, can_call_tools: bool):
    """按模型能力选择 Agent 路线。

    两条路线的差别在于"工具调用靠什么保证正确"：

      can_call_tools=True  -> 走 OpenAI tools 协议（create_tool_calling_agent）
          参数由服务端按 schema 产出并校验，模型不需要自己写对格式。
          代价：模型必须支持该协议。不支持时会被服务端直接拒绝（400）。

      can_call_tools=False -> 退回文本解析版 ReAct（build_react_agent）
          模型自己在正文里写 Thought / Action / Action Input，由解析器提取。
          好处是任何模型都能跑；代价是容易跑偏格式，所以要配
          handle_parsing_errors 让它自我纠正。
    """
    if can_call_tools:
        return create_tool_calling_agent(llm, tools, TOOL_CALLING_PROMPT)
    return build_react_agent(llm, tools, react_prompt)


def main() -> None:
    # 1. 读配置：provider 由本目录 .env 的 MODEL_PROVIDER 决定
    provider = get_provider()
    print("模型配置：", describe())

    # 2. 非云端模式下先探测服务，避免等到调用时才报错
    if provider not in ("huggingface",):
        ok, msg = check_server()
        print("服务探测：", "✅" if ok else "❌", msg)
        if not ok:
            raise SystemExit("服务未就绪，请先用 `python llm_config.py` 确认服务可用")

    llm = build_llm()

    # 2.5 先问一句"你到底支不支持 tools"，再决定走哪条路线。
    #     同一个 Ollama 上 deepseek-r1:32b 明确不支持、qwen3-vl:8b 支持，
    #     按名字猜迟早猜错，所以这里用实际探测。
    can_call_tools = supports_tools(llm)
    if can_call_tools:
        print("工具能力：✅ 支持 OpenAI tools 协议，走结构化工具调用路线")
    else:
        print("工具能力：⚠️  不支持 tools，退回文本解析版 ReAct（由提示词约束输出格式）")

    # 3. 组装工具
    tools = [
        StructuredTool(
            name="Search",
            func=safe_search,
            description="当你没有相关实时知识时，用于联网搜索最新信息",
            args_schema=SearchArgs,
        ),
        StructuredTool(
            name="CurrentTime",
            func=get_current_time,
            description="获取当前日期和时间，当你需要知道今天是什么时候时使用",
            args_schema=NoArgs,
        ),
    ]

    # 4. 提示词：优先读本地副本（官方 chat 版的逐字拷贝），它对模型的格式遵循度最好；
    #    本地副本缺失才联网 hub.pull，都失败才回落到本地中文 prompt
    prompt = build_prompt()

    # 5. 构建并执行 Agent
    #    build_react_agent 只在后端不支持 stop 参数时挂兼容解析器
    #    （本机 Ollama / 云端模型都原生支持 stop，此时与官方 create_react_agent 完全一致）
    #    max_iterations 控制"思考-行动"循环上限，防止小模型陷入死循环
    agent = build_agent(llm, tools, prompt, can_call_tools=can_call_tools)
    # handle_parsing_errors 只对文本解析那条路线有意义（模型写错格式时让它自我纠正）；
    # 走 tools 协议时不会触发，留着无害。
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors="你的输出不符合规定格式，请重新按 Question/Thought/Action 的格式输出。",
    )

    # 6. 多轮问答：把每轮的"问题 + 回答"追加进 chat_history，
    #    下一轮连同历史一起传给模型，它就记得上下文了
    #    （只记最终问答，中间的 Thought/Action 过程不进历史，避免污染上下文）
    chat_history: list[HumanMessage | AIMessage] = []
    needs_history = True if can_call_tools else "chat_history" in prompt.input_variables

    print("\n进入多轮问答（输入 exit 退出，clear 清空上下文）：")
    while True:
        try:
            question = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出")
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "退出"):
            break
        if question.lower() in ("clear", "清空"):
            chat_history.clear()
            print("已清空上下文")
            continue

        payload: dict = {"input": question}
        if needs_history:
            payload["chat_history"] = list(chat_history)

        result = agent_executor.invoke(payload)
        answer = result["output"]
        print("Agent：", answer)

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()
