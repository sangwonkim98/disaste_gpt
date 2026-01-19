"""
LangGraph 라우팅 로직

Agent 노드 이후 분기 결정을 담당합니다.
state의 next_action 값에 따라 다음 노드를 결정합니다.
"""

import logging
from typing import Literal
from langgraph.graph import END

from graph.state import GraphState

logger = logging.getLogger(__name__)


def route_agent(state: GraphState) -> Literal["tools", "retrieve", "report", "__end__"]:
    """
    Agent 노드 이후 분기 결정

    Args:
        state: 현재 그래프 상태

    Returns:
        다음 노드 이름 또는 END
    """
    next_action = state.get("next_action", "end")

    logger.info(f"🔀 [ROUTER] next_action: {next_action}")

    if next_action == "tools":
        logger.info("  → Tools 노드로 라우팅")
        return "tools"
    elif next_action == "retrieve":
        logger.info("  → Retrieve 노드로 라우팅")
        return "retrieve"
    elif next_action == "report":
        logger.info("  → Report 노드로 라우팅")
        return "report"
    else:
        logger.info("  → END로 라우팅")
        return END


def route_after_tools(state: GraphState) -> Literal["agent", "__end__"]:
    """
    Tools 노드 이후 분기 결정

    도구 실행 결과에 따라:
    - 추가 처리가 필요하면 Agent로 돌아감
    - 완료되면 END로 이동

    Args:
        state: 현재 그래프 상태

    Returns:
        다음 노드 이름
    """
    next_action = state.get("next_action", "end")

    if next_action == "agent":
        logger.info("🔀 [ROUTER] Tools 후 Agent로 돌아감")
        return "agent"
    else:
        logger.info("🔀 [ROUTER] Tools 후 END")
        return END


def route_after_retrieve(state: GraphState) -> Literal["agent", "__end__"]:
    """
    Retrieve 노드 이후 분기 결정

    RAG 검색 결과에 따라:
    - 추가 처리가 필요하면 Agent로 돌아감
    - 완료되면 END로 이동

    Args:
        state: 현재 그래프 상태

    Returns:
        다음 노드 이름
    """
    next_action = state.get("next_action", "end")

    if next_action == "agent":
        logger.info("🔀 [ROUTER] Retrieve 후 Agent로 돌아감")
        return "agent"
    else:
        logger.info("🔀 [ROUTER] Retrieve 후 END")
        return END
