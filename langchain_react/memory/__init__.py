"""记忆层 —— LangGraph 标准的两层记忆。

    checkpointer —— 短期记忆：thread 级，存 messages + slots，key = thread_id(=conversation_id)
    store        —— 长期记忆：跨 thread，存用户偏好 / 常用课题类型，namespace = ("user", user_id)

注意：业务任务状态（7022 的 session_id、7029 的 job_id）**不存这里**，
Agent 只存"指针"，内容留在各自服务内，避免两份状态互相冲突。
"""
