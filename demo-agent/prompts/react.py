"""官方 `hwchase17/react` ReAct 提示词的**本地副本**（单轮版）。

来源：https://smith.langchain.com/hub/hwchase17/react

与 `react_chat.py` 的区别：
    react      —— 没有 chat_history，用于单轮问答（ReAct.py）
    react_chat —— 多一段 Previous conversation history，用于多轮（ReAct_local.py）

为什么要落到本地：
    `hub.pull("hwchase17/react")` 走的是 api.smith.langchain.com，
    内网 / 断网 / 有代理约束的环境里它会失败。落盘之后，
    "能不能用上官方提示词"与网络状况彻底无关。

注意：这是**快照**，官方更新后不会自动同步，想同步见文件末尾的导出命令。
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate

# 官方原文，逐字照搬
_REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def load_react() -> PromptTemplate:
    """返回与 hub 上 `hwchase17/react` 等价的提示词对象。"""
    return PromptTemplate(
        input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
        template=_REACT_TEMPLATE,
    )


# ---------------------------------------------------------------------------
# 同步官方最新版（在有外网的环境执行一次，把输出粘回上面的 _REACT_TEMPLATE）：
#
#   python - <<'PY'
#   from langchain_classic import hub
#   print(hub.pull("hwchase17/react").template)
#   PY
# ---------------------------------------------------------------------------
