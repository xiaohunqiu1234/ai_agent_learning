"""qc_agent_mcp —— QC 智能体包（LangGraph 编排 + MCP 暴露）。

运行方式（均在仓库根目录执行，靠 PYTHONPATH 免安装）：

    export PYTHONPATH=$(pwd)
    python scripts/run_cli.py              # 命令行多轮调试（P1 冒烟入口）
    langgraph dev                          # Studio 本地可视化调试
    python -m apps.agent_api.main          # 7040 对话服务（SSE）
    python -m apps.mcp_server.main         # MCP Server（stdio / http）

目录遵循 LangGraph 官方约定：
    graph.py          —— 必须导出名为 graph 的编译后对象（langgraph.json 契约）
    state.py          —— TypedDict + reducer，可被 checkpointer 序列化
    configuration.py  —— 可调参数，由 config["configurable"] 注入
    nodes/ edges/     —— 节点与条件边分离，便于单测
    tools/            —— @tool + pydantic，docstring 即 description
    memory/           —— 短期(Checkpointer, thread 级) / 长期(Store, 跨 thread) 分离
    subgraphs/        —— 业务域子图（P3 落地）
"""

__version__ = "0.1.0"
