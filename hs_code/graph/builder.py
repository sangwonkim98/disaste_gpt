"""
LangGraph 그래프 빌더

그래프 구조를 정의하고 컴파일합니다.

그래프 구조:
START → Agent ─┬─→ Tools → END
               ├─→ Retrieve → END
               ├─→ Report → END
               └─→ END
"""

import logging
from typing import Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import GraphState
from graph.nodes import agent_node, tools_node, retrieve_node, report_node, summarize_node
from graph.router import route_agent

logger = logging.getLogger(__name__)


def build_graph(checkpointer: Optional[MemorySaver] = None):
    """
    LangGraph 그래프 빌드

    Args:
        checkpointer: 상태 체크포인터 (선택적, 멀티턴 대화 지원용)

    Returns:
        컴파일된 그래프
    """
    logger.info("🔨 LangGraph 그래프 빌드 시작...")

    # StateGraph 생성
    workflow = StateGraph(GraphState)

    # 노드 추가
    logger.info("  📦 노드 추가: agent, tools, retrieve, report, summarize")
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("report", report_node)
    workflow.add_node("summarize", summarize_node)

    # 진입점 설정
    workflow.set_entry_point("agent")

    # Agent 노드에서 조건부 분기
    logger.info("  🔀 조건부 엣지 추가: agent → [tools, retrieve, report, END]")
    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "retrieve": "retrieve",
            "report": "report",
            END: END
        }
    )

    # Tools -> Summarize -> END
    logger.info("  ➡️ 엣지 추가: tools → summarize → END")
    workflow.add_edge("tools", "summarize")
    workflow.add_edge("summarize", END)
    
    # Retrieve, Report는 END로 종료
    workflow.add_edge("retrieve", END)
    workflow.add_edge("report", END)

    # 그래프 컴파일
    if checkpointer:
        logger.info("  💾 체크포인터 사용: 멀티턴 대화 지원")
        graph = workflow.compile(checkpointer=checkpointer)
    else:
        graph = workflow.compile()

    logger.info("✅ LangGraph 그래프 빌드 완료")
    return graph


def build_graph_with_loop(checkpointer: Optional[MemorySaver] = None):
    """
    루프 지원 그래프 빌드 (Tools/Retrieve 후 Agent로 돌아가는 버전)

    그래프 구조:
    START → Agent ─┬─→ Tools → Agent (루프)
                   ├─→ Retrieve → Agent (루프)
                   ├─→ Report → END
                   └─→ END

    Args:
        checkpointer: 상태 체크포인터 (선택적)

    Returns:
        컴파일된 그래프
    """
    logger.info("🔨 LangGraph 그래프 빌드 (루프 버전) 시작...")

    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("report", report_node)

    # 진입점 설정
    workflow.set_entry_point("agent")

    # Agent 노드에서 조건부 분기
    workflow.add_conditional_edges(
        "agent",
        route_agent,
        {
            "tools": "tools",
            "retrieve": "retrieve",
            "report": "report",
            END: END
        }
    )

    # Tools, Retrieve는 Agent로 돌아감 (루프)
    workflow.add_edge("tools", "agent")
    workflow.add_edge("retrieve", "agent")

    # Report는 바로 종료
    workflow.add_edge("report", END)

    # 그래프 컴파일
    if checkpointer:
        graph = workflow.compile(checkpointer=checkpointer)
    else:
        graph = workflow.compile()

    logger.info("✅ LangGraph 그래프 빌드 (루프 버전) 완료")
    return graph


# 기본 그래프 인스턴스 (지연 초기화)
_default_graph = None


def get_default_graph():
    """기본 그래프 싱글톤 반환"""
    global _default_graph
    if _default_graph is None:
        _default_graph = build_graph()
    return _default_graph


def get_graph_with_memory():
    """메모리 체크포인터가 있는 그래프 반환"""
    checkpointer = MemorySaver()
    return build_graph(checkpointer=checkpointer)
