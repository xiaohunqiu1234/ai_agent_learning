"""工具层 —— 把现有 HTTP 服务包装成 LangChain/LangGraph 可用的 Tool。

四条封装契约（缺一不可，否则 ReAct 会被脏 Observation 带偏）：
    1. pydantic args_schema，字段一律给默认值 —— 模型漏填或给空字典时不能崩
    2. 异常不抛出，转成错误文本返回 —— 让 Agent 把失败当成一次 Observation 继续推理
    3. 返回值截断 + 结构化 —— 长文本先截断再回灌，防止打爆上下文
    4. 统一超时 / 重试 / request_id / 日志
"""
