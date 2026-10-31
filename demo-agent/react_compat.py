"""针对"不尊重 stop 参数"的模型的 ReAct 兼容层（默认启用）。

问题现象：
    标准 ReAct 依赖 stop=["Observation:"] 让模型在输出 Action 后就停住，
    由 AgentExecutor 去执行工具、再把**真实的** Observation 拼回去。
    但内网 / 自建网关常常直接忽略 stop（实测本项目的 LLM 代理服务也不支持），
    模型于是一路替你把 Observation 和 Final Answer 都编出来了。

后果：
    ReActSingleInputOutputParser 只要看到 "Final Answer:" 就返回 AgentFinish，
    所以工具根本不会被执行，答案全是模型自己编的。

对策：
    在交给标准解析器之前，先把模型自撰的 Observation 及之后的内容裁掉，
    只保留 Question/Thought/Action(+Action Input)，从而触发真正的工具调用。

用法：
    from react_compat import build_react_agent
    agent = build_react_agent(llm, tools, prompt)
"""

from __future__ import annotations

import os
import re

from langchain_classic.agents import create_react_agent
from langchain_classic.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.tools import BaseTool

# 中英文冒号都覆盖；只切第一段，保留 Question/Thought/Action 部分
_HALLUCINATED_OBSERVATION = re.compile(r"Observation\s*[:：]")


class StopAwareReActParser(ReActSingleInputOutputParser):
    """兼容"不尊重 stop 参数"模型的 ReAct 解析器。

    它接管三类标准解析器处理不了的情况：

    1. 模型自撰 Observation
       stop 失效时模型会一路替你把 Observation 写下去。这里按 Action 取用，
       在"自撰的 Observation"处截断，避免把幻觉内容当成工具入参。
       注意：**只在确实存在 Action 时才截断**，否则最终答案里若出现 "Observation:"
       字样会被误裁。

    2. 同时出现 Action 与 Final Answer
       标准实现只要看到 "Final Answer:" 就返回 AgentFinish，于是一次工具都不调。
       这里反过来：只要有可解析的 Action，就**优先按调工具处理**。

    3. 模型彻底丢格式改说散文
       既没有 Action 也没有 Final Answer 时，标准实现抛 OutputParserException，
       AgentExecutor 会拿 handle_parsing_errors 的提示重试，但模型往往重复同样的散文，
       直到耗尽 max_iterations 报 "Agent stopped due to iteration limit"。
       这里直接把这段散文当作最终答案收尾，让链路正常结束。

    除此之外（模型规规矩矩输出），行为与官方解析器完全一致。
    """

    _ACTION_REGEX = re.compile(
        r"Action\s*\d*\s*:[\s]*(.*?)Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", re.DOTALL
    )

    @staticmethod
    def _clean(value: str) -> str:
        return value.strip().strip("`").strip().strip('"').strip()

    def parse(self, text: str):  # type: ignore[override]
        # ① 有工具调用意图 -> 优先执行工具
        match = self._ACTION_REGEX.search(text or "")
        if match:
            # 裁掉自撰 Observation 之后的内容（含幻觉的 Observation / Final Answer）
            before_observation = _HALLUCINATED_OBSERVATION.split(text, maxsplit=1)[0]
            usable = self._ACTION_REGEX.search(before_observation) or match
            return AgentAction(
                self._clean(usable.group(1)), self._clean(usable.group(2)), text
            )

        # ② 没有工具意图 -> 交给标准解析器处理 Final Answer
        try:
            return super().parse(text)
        except OutputParserException:
            # ③ 既没 Action 也没 Final Answer：把散文当最终答案，避免卡在重试循环
            cleaned = (text or "").strip()
            if not cleaned:
                raise
            return AgentFinish({"output": cleaned}, text)


# 宽松收尾已并入 StopAwareReActParser，保留旧名字以兼容既有导入
LenientReActParser = StopAwareReActParser


def ignore_stop() -> bool:
    """是否启用兼容解析器，**默认开启**。

    默认开启的理由：兼容解析器在模型守规矩时与官方解析器行为完全一致（见
    StopAwareReActParser 的说明），只在模型不守规矩时才介入；而"该不该介入"很难
    靠 provider 名字猜对——云端模型也可能忽略 stop，本地服务也可能支持得很好。

    实测踩过的坑：provider=ollama 时旧版判定为"支持 stop"而退回官方解析器，
    结果模型照样把 Observation 和 Final Answer 一起写出来，直接抛
    "Parsing LLM output produced both a final answer and a parse-able action"。

    确实不需要时（想严格用官方行为）用 LOCAL_IGNORE_STOP=0 关掉即可。
    """
    raw = os.getenv("LOCAL_IGNORE_STOP")
    if raw is None:
        return True
    return raw.strip().lower() in ("1", "true", "yes")


def build_react_agent(
    llm: BaseLanguageModel,
    tools: list[BaseTool],
    prompt: BasePromptTemplate,
    **kwargs,
):
    """构建 ReAct Agent，按需挂上兼容解析器。"""
    if ignore_stop():
        # Lenient 版额外容忍"模型彻底丢格式直接散文作答"的情况，避免反复重试到迭代上限
        kwargs["output_parser"] = LenientReActParser()
    return create_react_agent(llm, tools, prompt, **kwargs)
