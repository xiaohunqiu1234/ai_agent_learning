"""长期记忆：跨会话（跨 thread）的用户级记忆。

P1 先用 InMemoryStore 占位；P3 再接入真实读写
（保存用户常用课题类型、常用单位名、历史附件等），
届时换 PostgresStore 即可，调用代码不变。
"""

from __future__ import annotations

from typing import Any


def build_store():
    from langgraph.store.memory import InMemoryStore

    return InMemoryStore()


def user_namespace(user_id: str) -> tuple[str, str]:
    """长期记忆的命名空间：同一用户的多个会话共享。"""
    return ("user", user_id or "anonymous")


def remember(store, user_id: str, key: str, value: Any) -> None:
    store.put(user_namespace(user_id), key, value)


def recall(store, user_id: str, key: str) -> Any:
    item = store.get(user_namespace(user_id), key)
    return None if item is None else item.value
