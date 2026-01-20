"""
Query Pipeline 그래프 빌더

그래프 구조:
START → analyzer ─┬─→ [executor 그룹] → synthesizer → END
                  └─→ direct_response → END

executor 그룹 (병렬 실행 가능):
  - tool_executor (API 호출)
  - rag_executor (매뉴얼 검색)
  - pdf_executor (PDF 분석)
"""

import logging
from typing import Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.query_state import QueryState
from graph.query_nodes import (
    analyzer_node,
    tool_executor_node,
    rag_executor_node,
    pdf_executor_node,
    synthesizer_node,
    direct_response_node
)

logger = logging.getLogger(__name__)


def route_after_analyzer(state: QueryState) -> Literal["executor", "direct_response", "end"]:
    """
    Analyzer 노드 이후 라우팅 결정

    - executor: 도구/RAG/PDF 실행 필요
    - direct_response: 직접 응답 (일반 대화)
    - end: 즉시 종료 (에러 등)
    """
    next_node = state.get("next_node", "end")

    if next_node == "executor":
        plan = state.get("execution_plan")
        if plan and (plan.get("need_tools") or plan.get("need_rag") or plan.get("need_pdf")):
            return "executor"
        return "direct_response"

    elif next_node == "direct_response":
        return "direct_response"

    else:
        return "end"


def route_after_executor(state: QueryState) -> Literal["synthesizer", "end"]:
    """
    Executor 노드들 이후 라우팅 결정
    항상 synthesizer로 이동 (결과 종합)
    """
    return "synthesizer"


def combined_executor_node(state: QueryState) -> dict:
    """
    모든 Executor를 순차 실행하는 통합 노드

    LangGraph의 기본 구조에서는 병렬 실행이 복잡하므로,
    여기서는 순차 실행으로 구현 (필요시 asyncio로 병렬화 가능)
    """
    logger.info("⚡ [COMBINED_EXECUTOR] 실행 시작...")

    plan = state.get("execution_plan")
    if not plan:
        return {}

    result = {}

    # 1. Tool Executor
    if plan.get("need_tools"):
        tool_output = tool_executor_node(state)
        if tool_output.get("execution_result"):
            result = tool_output["execution_result"]
            # state 업데이트 시뮬레이션
            state["execution_result"] = result

    # 2. RAG Executor
    if plan.get("need_rag"):
        rag_output = rag_executor_node(state)
        if rag_output.get("execution_result"):
            result.update(rag_output["execution_result"])
            state["execution_result"] = result

    # 3. PDF Executor
    if plan.get("need_pdf"):
        pdf_output = pdf_executor_node(state)
        if pdf_output.get("execution_result"):
            result.update(pdf_output["execution_result"])

    logger.info("✅ [COMBINED_EXECUTOR] 모든 실행 완료")

    return {
        "execution_result": result,
        "next_node": "synthesizer"
    }


def build_query_graph(checkpointer: Optional[MemorySaver] = None):
    """
    Query Pipeline 그래프 빌드

    구조:
    ```
    START
      │
      ▼
    analyzer (의도 분석 + 계획 수립)
      │
      ├─[need_tools/rag/pdf]──▶ executor ──▶ synthesizer ──▶ END
      │
      └─[일반 대화]──▶ direct_response ──▶ END
    ```
    """
    logger.info("🔨 Query Pipeline 그래프 빌드 시작...")

    # StateGraph 생성
    workflow = StateGraph(QueryState)

    # 노드 추가
    logger.info("  📦 노드 추가: analyzer, executor, synthesizer, direct_response")
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("executor", combined_executor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("direct_response", direct_response_node)

    # 진입점 설정
    workflow.set_entry_point("analyzer")

    # Analyzer 이후 조건부 분기
    logger.info("  🔀 조건부 엣지: analyzer → [executor, direct_response, END]")
    workflow.add_conditional_edges(
        "analyzer",
        route_after_analyzer,
        {
            "executor": "executor",
            "direct_response": "direct_response",
            "end": END
        }
    )

    # Executor → Synthesizer → END
    logger.info("  ➡️ 엣지: executor → synthesizer → END")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", END)

    # Direct Response → END
    logger.info("  ➡️ 엣지: direct_response → END")
    workflow.add_edge("direct_response", END)

    # 그래프 컴파일
    if checkpointer:
        logger.info("  💾 체크포인터 사용: 멀티턴 대화 지원")
        graph = workflow.compile(checkpointer=checkpointer)
    else:
        graph = workflow.compile()

    logger.info("✅ Query Pipeline 그래프 빌드 완료")

    # 그래프 구조 시각화 (ASCII)
    _log_graph_structure()

    return graph


def _log_graph_structure():
    """그래프 구조 ASCII 출력"""
    structure = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                    QUERY PIPELINE GRAPH                       ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║                                                               ║
    ║   START                                                       ║
    ║     │                                                         ║
    ║     ▼                                                         ║
    ║  ┌──────────┐                                                 ║
    ║  │ ANALYZER │  의도 분석 + 실행 계획 수립                       ║
    ║  └────┬─────┘                                                 ║
    ║       │                                                       ║
    ║       ├─── need_tools/rag/pdf ───┐                            ║
    ║       │                          ▼                            ║
    ║       │                   ┌──────────┐                        ║
    ║       │                   │ EXECUTOR │  도구/RAG/PDF 실행      ║
    ║       │                   └────┬─────┘                        ║
    ║       │                        │                              ║
    ║       │                        ▼                              ║
    ║       │                   ┌────────────┐                      ║
    ║       │                   │SYNTHESIZER │  결과 종합 → 응답     ║
    ║       │                   └────┬───────┘                      ║
    ║       │                        │                              ║
    ║       │                        ▼                              ║
    ║       │                       END                             ║
    ║       │                                                       ║
    ║       └─── 일반 대화 ───┐                                      ║
    ║                         ▼                                     ║
    ║                  ┌───────────────┐                            ║
    ║                  │DIRECT_RESPONSE│  LLM 직접 응답              ║
    ║                  └───────┬───────┘                            ║
    ║                          │                                    ║
    ║                          ▼                                    ║
    ║                         END                                   ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    logger.info(structure)


# ===== 편의 함수 =====

_default_query_graph = None


def get_query_graph():
    """Query Pipeline 그래프 싱글톤 반환"""
    global _default_query_graph
    if _default_query_graph is None:
        _default_query_graph = build_query_graph()
    return _default_query_graph


def get_query_graph_with_memory():
    """메모리 체크포인터가 있는 Query 그래프 반환"""
    checkpointer = MemorySaver()
    return build_query_graph(checkpointer=checkpointer)


# ===== 테스트 함수 =====

def test_query_pipeline():
    """Query Pipeline 테스트"""
    print("\n" + "="*60)
    print("🧪 Query Pipeline 테스트")
    print("="*60)

    graph = get_query_graph()

    test_cases = [
        {"user_input": "서울 현재 날씨 알려줘", "debug_mode": True},
        {"user_input": "호우 특보 시 대응 절차가 뭐야?", "debug_mode": True},
        {"user_input": "지금 기상특보 현황이랑 대응 매뉴얼 같이 알려줘", "debug_mode": True},
        {"user_input": "안녕하세요", "debug_mode": True},
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- 테스트 {i}: {test_input['user_input'][:30]}... ---")

        initial_state = {
            "messages": [],
            "user_input": test_input["user_input"],
            "debug_mode": test_input.get("debug_mode", False),
            "reasoning_mode": False,  # 테스트용으로 비활성화
            "uploaded_pdf_content": None,
            "selected_manual": None,
        }

        try:
            result = graph.invoke(initial_state)
            print(f"✅ 실행 계획: {result.get('execution_plan', {})}")
            print(f"📝 응답 (첫 200자): {result.get('final_response', '')[:200]}...")
        except Exception as e:
            print(f"❌ 에러: {e}")

    print("\n" + "="*60)
    print("🧪 테스트 완료")
    print("="*60)


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    test_query_pipeline()
