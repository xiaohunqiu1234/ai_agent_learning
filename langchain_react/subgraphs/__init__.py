"""业务域子图（P3 落地）。

规划：
    knowledge_qa.py         —— 知识检索问答（7026 + Dify 知识库）
    report_generation.py    —— 报告生成（7022，submit → poll 两节点）
    qc_review.py            —— QC 审查（7025）
    report_optimization.py  —— 报告优化（7029）

为什么拆子图而不是塞工具：
    这四个域都是"分钟级长任务 + 多步编排"，塞进单条 ReAct 循环会让
    工具列表膨胀、模型选错工具。子图可以把域内流程固化下来，
    主图只做路由（supervisor 模式），既可控又能独立调试。
"""
