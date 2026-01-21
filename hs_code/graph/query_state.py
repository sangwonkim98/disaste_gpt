"""
Query Pipeline State 정의

정보 조회 파이프라인을 위한 상태 스키마
- tools, rag, pdf_reader 병렬 실행 지원
- 도구 선택 reasoning 포함
"""

from typing import TypedDict, Annotated, List, Optional, Any, Dict
from langgraph.graph.message import add_messages


class ExecutionPlan(TypedDict):
    """실행 계획"""
    need_tools: bool                   # 외부 API 호출 필요
    need_rag: bool                     # 매뉴얼 RAG 검색 필요
    need_pdf: bool                     # 사용자 업로드 PDF 분석 필요

    tool_list: List[str]               # 호출할 도구 목록
    tool_params: Dict[str, Dict]       # 도구별 파라미터
    tool_reasoning: str                # 도구 선택 이유

    rag_query: Optional[str]           # RAG 검색 쿼리 (재구성된)
    rag_reasoning: Optional[str]       # RAG 필요 이유

    pdf_task: Optional[str]            # PDF 분석 작업 유형

    confidence: float                  # 계획 확신도 (0.0 ~ 1.0)


class ExecutionResult(TypedDict):
    """실행 결과"""
    tool_results: Optional[Dict[str, Any]]   # 도구 실행 결과
    rag_results: Optional[List[Dict]]        # RAG 검색 결과
    pdf_results: Optional[str]               # PDF 분석 결과


class QueryState(TypedDict):
    """Query Pipeline 상태"""

    # ===== 기본 필드 =====
    messages: Annotated[list, add_messages]  # 대화 히스토리
    user_input: str                          # 현재 사용자 입력

    # ===== 컨텍스트 =====
    uploaded_pdf_content: Optional[str]      # 업로드된 PDF 내용
    selected_manual: Optional[str]           # 선택된 매뉴얼 (RAG용)

    # ===== 설정 =====
    debug_mode: bool                         # 디버그 로깅 활성화
    reasoning_mode: bool                     # 추론 모드 (extended thinking)

    # ===== Analyzer 출력 =====
    execution_plan: Optional[ExecutionPlan]  # 실행 계획

    # ===== Executor 출력 =====
    execution_result: Optional[ExecutionResult]  # 실행 결과

    # ===== Synthesizer 출력 =====
    final_response: str                      # 최종 응답

    # ===== Analyzer 스트리밍 =====
    ready_to_analyze: bool                   # Analyzer 스트리밍 준비 신호
    analyzer_messages: List[Any]             # Analyzer LLM에 보낼 메시지

    # ===== Synthesizer 스트리밍 =====
    ready_to_generate: bool                  # 최종 응답 스트리밍 준비 신호
    final_messages: List[Any]                # LLM에 보낼 최종 메시지 리스트

    # ===== 라우팅 =====
    next_node: str                           # 다음 노드 ("executor" | "direct_response" | "end")


# ===== 도구 메타데이터 =====
TOOL_METADATA = {
    "serpapi_web_search": {
        "category": "search",
        "description": "최신 뉴스, 사건 정보 검색",
        "keywords": ["뉴스", "검색", "최근", "사건", "소식"],
        "requires_location": False,
    },
    "kma_get_ultra_srt_ncst": {
        "category": "weather_current",
        "description": "현재 실시간 날씨 (기온, 강수, 바람)",
        "keywords": ["현재 날씨", "지금 날씨", "실시간", "기온", "온도"],
        "requires_location": True,
    },
    "kma_get_ultra_srt_fcst": {
        "category": "weather_forecast",
        "description": "초단기예보 (6시간)",
        "keywords": ["초단기", "6시간"],
        "requires_location": True,
    },
    "kma_get_vilage_fcst": {
        "category": "weather_forecast",
        "description": "단기예보 (3일, 3시간 간격)",
        "keywords": ["예보", "내일", "모레", "단기"],
        "requires_location": True,
    },
    "kma_get_mid_land_fcst": {
        "category": "weather_midterm",
        "description": "중기육상예보 (3~10일)",
        "keywords": ["주간", "이번주", "중기", "장기"],
        "requires_location": True,
    },
    "kma_get_mid_ta": {
        "category": "weather_midterm",
        "description": "중기기온예보 (3~10일 최저/최고)",
        "keywords": ["주간 기온", "이번주 기온"],
        "requires_location": True,
    },
    "kma_get_pwn_status": {
        "category": "warning",
        "description": "예비특보 현황",
        "keywords": ["예비특보", "예비"],
        "requires_location": False,
    },
    "kma_get_wthr_wrn_msg": {
        "category": "warning",
        "description": "기상특보 (태풍, 호우, 폭염 등)",
        "keywords": ["특보", "경보", "주의보", "태풍", "호우", "폭염", "한파", "대설"],
        "requires_location": False,
    },
    "kma_get_eqk_msg_list": {
        "category": "earthquake",
        "description": "최근 지진 목록",
        "keywords": ["지진", "earthquake", "지진 목록"],
        "requires_location": False,
    },
    "kma_get_eqk_msg": {
        "category": "earthquake",
        "description": "지진 상세 정보",
        "keywords": ["지진 상세"],
        "requires_location": False,
    },
}

# ===== 도구 의존성 (자동 체이닝) =====
TOOL_DEPENDENCIES = {
    # 지진 목록 조회 시 → 상세 정보도 함께 (선택적)
    "kma_get_eqk_msg_list": {
        "suggests": ["kma_get_eqk_msg"],
        "auto_chain": False,  # 목록에 결과가 있을 때만
        "reason": "지진 목록에서 상세 정보를 가져올 수 있습니다"
    },
    # 중기육상예보 → 중기기온예보 함께
    "kma_get_mid_land_fcst": {
        "suggests": ["kma_get_mid_ta"],
        "auto_chain": True,
        "reason": "중기 날씨와 기온 정보를 함께 제공합니다"
    },
}

# ===== 키워드 → 도구 매핑 (Rule-based fallback) =====
KEYWORD_TOOL_MAP = {
    # 현재 날씨
    "현재 날씨": ["kma_get_ultra_srt_ncst"],
    "지금 날씨": ["kma_get_ultra_srt_ncst"],
    "실시간 날씨": ["kma_get_ultra_srt_ncst"],
    "기온": ["kma_get_ultra_srt_ncst"],
    "온도": ["kma_get_ultra_srt_ncst"],

    # 예보
    "날씨 예보": ["kma_get_vilage_fcst"],
    "내일 날씨": ["kma_get_vilage_fcst"],
    "모레 날씨": ["kma_get_vilage_fcst"],
    "주간 예보": ["kma_get_mid_land_fcst", "kma_get_mid_ta"],
    "이번주 날씨": ["kma_get_mid_land_fcst", "kma_get_mid_ta"],

    # 특보/경보
    "특보": ["kma_get_wthr_wrn_msg"],
    "기상특보": ["kma_get_wthr_wrn_msg"],
    "경보": ["kma_get_wthr_wrn_msg"],
    "주의보": ["kma_get_wthr_wrn_msg"],
    "태풍": ["kma_get_wthr_wrn_msg", "serpapi_web_search"],
    "호우": ["kma_get_wthr_wrn_msg"],
    "폭염": ["kma_get_wthr_wrn_msg"],
    "한파": ["kma_get_wthr_wrn_msg"],
    "대설": ["kma_get_wthr_wrn_msg"],

    # 지진
    "지진": ["kma_get_eqk_msg_list"],
    "earthquake": ["kma_get_eqk_msg_list"],

    # 검색
    "뉴스": ["serpapi_web_search"],
    "검색": ["serpapi_web_search"],
    "최근 소식": ["serpapi_web_search"],
}

# ===== RAG 키워드 =====
RAG_KEYWORDS = [
    "매뉴얼", "규정", "지침", "절차", "가이드", "방법",
    "행동요령", "대응", "조치", "기준", "정의",
    "어떻게", "무엇", "뭐야", "알려줘", "설명",
    "풍수해", "재난", "재해", "대피"
]
