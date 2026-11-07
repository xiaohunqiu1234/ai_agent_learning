"""文本处理：截断与格式纠错提示。

截断是硬需求：审查/检索返回的 Observation 动辄几万字，
不截断会打爆 7888 的上下文窗口，后续轮次直接退化。
"""

from __future__ import annotations


def truncate(text: str, limit: int = 3000, tail_hint: str = "\n…（已截断）") -> str:
    """按字符数截断，保留结尾提示，让模型知道内容不完整。"""
    if text is None:
        return ""
    text = str(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + tail_hint


def strip_code_fence(text: str) -> str:
    """剥掉模型爱加的 ```json … ``` 代码块。"""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
    return s
