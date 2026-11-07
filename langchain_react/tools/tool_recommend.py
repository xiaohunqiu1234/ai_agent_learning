"""7001 管理工具智能推荐服务 → LangChain 工具。

接口前缀：/qc/ai/tool_recommend（端口 7001）
    /recommend/ai   —— 全链路推荐（项目分类 + 项目类型 + 活动节点 → 推荐工具清单）
    /enhance/table  —— 阶段映射表查询（活动节点 → 优先/次推工具）
    /retrieve/tool  —— 工具详情知识库检索
    /llm/intent     —— LLM 意图识别
"""

from __future__ import annotations

from langchain_core.tools import tool

from qc_agent_mcp.services import service_url
from qc_agent_mcp.tools.http_client import post_json, safe_call, short

_TIMEOUT = 120.0


@tool
def recommend_ai(
    project_category: str = "",
    project_type: str = "",
    activity_node_name: str = "",
) -> str:
    """按项目信息推荐适用管理工具（7001 全链路推荐，只读）。

    输入项目分类、项目类型、活动节点名称三项；返回推荐工具清单及来源。
    当用户问"这个阶段该用什么管理工具"时使用。
    """
    def _call():
        return post_json(
            service_url("tool_recommend", "/qc/ai/tool_recommend/recommend/ai"),
            payload={
                "project_category": project_category,
                "project_type": project_type,
                "activity_node_name": activity_node_name,
            },
            timeout=_TIMEOUT,
        )
    return short(safe_call(_call), 2000)


@tool
def enhance_table(activity_name: str = "") -> str:
    """查询活动节点的阶段映射表（7001，只读）。

    输入活动节点名称，返回该节点的优先工具（priority_tools）与次推工具（fallback_tools）。
    推荐工具前想先看某节点默认配置时使用。
    """
    def _call():
        return post_json(
            service_url("tool_recommend", "/qc/ai/tool_recommend/enhance/table"),
            payload={"activity_name": activity_name},
            timeout=_TIMEOUT,
        )
    return short(safe_call(_call), 2000)


@tool
def retrieve_tool_detail(search_query: str = "") -> str:
    """在管理工具详情知识库中检索（7001，只读）。

    输入检索关键词（建议用纯关键词而非长句），返回命中的工具知识条目。
    需要了解某个管理工具的详情/用法时使用。
    """
    def _call():
        return post_json(
            service_url("tool_recommend", "/qc/ai/tool_recommend/retrieve/tool"),
            payload={"search_query": search_query},
            timeout=_TIMEOUT,
        )
    return short(safe_call(_call), 3000)


@tool
def tool_intent(query: str = "") -> str:
    """对用户问题做管理工具场景的 LLM 意图识别（7001，只读）。

    输入用户问题，返回识别出的意图结构化结果。与 7028 intent_judgment 的区别：
    本工具专门面向"管理工具推荐"场景。
    """
    def _call():
        return post_json(
            service_url("tool_recommend", "/qc/ai/tool_recommend/llm/intent"),
            payload={"query": query},
            timeout=_TIMEOUT,
        )
    return short(safe_call(_call), 2000)
