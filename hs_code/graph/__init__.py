"""
LangGraph 기반 그래프 모듈

이 패키지는 LangGraph를 사용한 에이전트 그래프를 정의합니다.

주요 컴포넌트:
- state.py: GraphState 정의 (기존)
- nodes.py: 노드 함수들 (agent, tools, retrieve, report) (기존)
- router.py: 라우팅 로직 (기존)
- builder.py: 그래프 빌드 및 컴파일 (기존)

Query Pipeline (v2):
- query_state.py: QueryState 정의
- query_nodes.py: 분석/실행/종합 노드
- query_builder.py: Query Pipeline 그래프 빌드
- debug_utils.py: 디버깅 유틸리티
"""

# 기존 그래프 (하위 호환성)
from graph.state import GraphState
from graph.builder import build_graph, get_default_graph, get_graph_with_memory

# Query Pipeline (v2)
from graph.query_state import QueryState, ExecutionPlan, ExecutionResult
from graph.query_builder import build_query_graph, get_query_graph, get_query_graph_with_memory

__all__ = [
    # 기존
    "GraphState",
    "build_graph",
    "get_default_graph",
    "get_graph_with_memory",
    # Query Pipeline v2
    "QueryState",
    "ExecutionPlan",
    "ExecutionResult",
    "build_query_graph",
    "get_query_graph",
    "get_query_graph_with_memory",
]
