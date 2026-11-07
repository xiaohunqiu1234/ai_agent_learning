"""短期记忆：thread 级会话持久化。

thread_id 由上层传入的 conversation_id 充当——同一个 conversation_id
多轮调用就自动续接上下文与槽位，进程重启也不丢。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_DB = "var/state/qc_agent_mcp.sqlite"


def build_checkpointer():
    """按 QC_AGENT_CHECKPOINTER 选择实现。

    sqlite（默认）：零额外组件，重启不丢，够当前单机部署。
    memory      ：仅调试用，进程退出即丢。
    未来多副本部署换 PostgresSaver / RedisSaver，接口不变，只改这里。
    """
    mode = os.getenv("QC_AGENT_CHECKPOINTER", "sqlite").lower()

    if mode == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()

    db_path = os.getenv("QC_AGENT_STATE_DB", DEFAULT_DB)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver.from_conn_string(db_path)
    except ImportError:
        # 缺 langgraph-checkpoint-sqlite 时退化成内存态，保证图仍能起来（调试用）
        print(
            f"[提示] 未安装 langgraph-checkpoint-sqlite，改用内存态 Checkpointer"
            f"（进程重启即丢）。如需持久化：pip install langgraph-checkpoint-sqlite",
            file=sys.stderr,
        )
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver()
