"""
LangGraph State 정의

GraphState는 그래프 실행 중 모든 노드 간에 공유되는 상태를 정의합니다.
"""

from typing import TypedDict, Annotated, List, Optional, Any
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """LangGraph 그래프 상태 정의"""

    # 메시지 히스토리 (LangGraph 기본 - add_messages로 누적)
    messages: Annotated[list, add_messages]

    # 사용자 입력
    user_input: str
    uploaded_context: Optional[str]  # PDF/TXT 업로드 내용

    # 설정 옵션
    agent_mode: bool          # Agent 모드 활성화 여부
    reasoning_mode: bool      # 추론 모드 활성화 여부
    enable_rag: bool          # RAG 활성화 여부
    selected_pdf: Optional[str]  # 선택된 PDF 경로

    # Agent 판단 결과
    next_action: str  # "tools" | "retrieve" | "report" | "end"

    # 각 노드 결과
    tool_results: Optional[str]      # Tools 노드 실행 결과
    rag_results: Optional[str]       # RAG 검색 결과
    report_output: Optional[str]     # Report 생성 결과

    # 에이전트 내부 상태
    tool_calls: Optional[List[Any]]  # 실행할 도구 호출 목록
    thinking_content: Optional[str]  # 추론 과정 내용

    # 최종 응답
    final_response: str

    # 참조 문서 정보
    reference_docs: Optional[str]    # 참조 문서 표시 정보
