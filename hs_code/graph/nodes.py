"""
LangGraph 노드 함수 정의

각 노드는 기존 서비스 모듈을 래핑하여 LangGraph 상태 기반으로 동작합니다.
- agent_node: LLM 호출 및 다음 행동 판단
- tools_node: 외부 API 도구 실행 (날씨, 검색 등)
- retrieve_node: RAG 검색 실행
- report_node: 보고서 생성 파이프라인
"""

import logging
import json
import re
from typing import Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import VLLM_SERVER_URL, VLLM_API_KEY, LLM_MODEL_NAME
from graph.state import GraphState

logger = logging.getLogger(__name__)

# 지연 로딩을 위한 전역 변수
_agent_manager = None
_agent_tools = None
_rag_system = None
_report_generator = None
_llm_client = None


def _get_agent_manager():
    """ExaoneAgentManager 싱글톤 반환"""
    global _agent_manager
    if _agent_manager is None:
        from core.agent_manager import ExaoneAgentManager
        _agent_manager = ExaoneAgentManager()
    return _agent_manager


def _get_agent_tools():
    """ExaoneAgentTools 싱글톤 반환"""
    global _agent_tools
    if _agent_tools is None:
        from services.agent_tools import exaone_agent_tools
        _agent_tools = exaone_agent_tools
    return _agent_tools


def _get_rag_system():
    """AdvancedRAGSystem 싱글톤 반환"""
    global _rag_system
    if _rag_system is None:
        from services.rag_engine import AdvancedRAGSystem
        _rag_system = AdvancedRAGSystem()
    return _rag_system


def _get_report_generator():
    """ReportGenerator 싱글톤 반환"""
    global _report_generator
    if _report_generator is None:
        from core.generator import ReportGenerator
        _report_generator = ReportGenerator()
    return _report_generator


def _get_llm_client():
    """LLM 클라이언트 싱글톤 반환"""
    global _llm_client
    if _llm_client is None:
        from core.llm_client import ExaoneClient
        _llm_client = ExaoneClient(
            server_url=VLLM_SERVER_URL,
            api_key=VLLM_API_KEY,
            model_name=LLM_MODEL_NAME
        )
    return _llm_client


def _convert_messages_to_history(messages: List) -> List[List[str]]:
    """LangGraph 메시지 리스트를 기존 history 형식으로 변환"""
    history = []
    current_user = None

    def _ensure_string(content):
        """content가 list인 경우 문자열로 변환"""
        if isinstance(content, list):
            return " ".join(str(c) for c in content)
        return str(content) if content else ""

    for msg in messages:
        if isinstance(msg, HumanMessage):
            current_user = _ensure_string(msg.content)
        elif isinstance(msg, AIMessage):
            if current_user is not None:
                history.append([current_user, _ensure_string(msg.content)])
                current_user = None

    return history


def _detect_intent_with_llm(user_input: str) -> str:
    """
    [LLM 기반 의도 파악]
    LLM의 추론 능력을 사용하여 문맥에 맞는 정확한 의도를 분류합니다.
    """
    ROUTER_SYSTEM_PROMPT = """
당신은 사용자 질문의 의도를 분류하는 'Intent Classifier'입니다.
사용자의 입력을 분석하여 다음 4가지 카테고리 중 하나를 선택해 JSON 형식으로 답변하세요.

1. "report": 보고서 작성, 문서 생성 요청
2. "tools": 날씨, 실시간 정보, 뉴스 검색 등 외부 데이터 필요
3. "retrieve": 매뉴얼, 규정, 가이드, 사내 지침 등 내부 지식 검색
4. "end": 일반적인 대화, 인사, 작별 인사

**출력 형식:**
{"intent": "선택한_카테고리", "reason": "선택 이유"}

**주의사항:**
- 오직 JSON 데이터만 출력하세요.
- 불필요한 설명은 제외하세요.
"""

    try:
        client = _get_llm_client()

        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]

        response = client.generate_response(
            messages=messages,
            temperature=0.0, 
            enable_thinking=False, 
            stream=False
        )

        content = response.choices[0].message.content.strip()
        
        if "```" in content:
            if "json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            else:
                content = content.split("```")[1].strip()

        result = json.loads(content)
        intent = result.get("intent", "end")

        logger.info(f"🧠 [LLM ROUTER] 판단: {intent} (이유: {result.get('reason')})")
        
        valid_intents = ["report", "tools", "retrieve", "end"]
        if intent not in valid_intents:
            return "end"
            
        return intent

    except Exception as e:
        logger.error(f"❌ [LLM ROUTER] 라우팅 실패 (Rule-based로 넘어감): {e}")
        return _detect_intent_legacy(user_input)


def _detect_intent_legacy(user_input: str) -> str:
    """
    (Legacy) 사용자 입력에서 의도를 파악하여 다음 행동 결정
    """
    user_input_lower = user_input.lower()

    # 보고서 생성 관련 키워드
    report_keywords = [
        "보고서", "report", "일일보고", "상황보고",
        "문서 생성", "작성해줘", "생성해줘"
    ]
    for kw in report_keywords:
        if kw in user_input_lower:
            return "report"

    # 도구 사용이 필요한 키워드 (날씨, 실시간 정보)
    tool_keywords = [
        "날씨", "기온", "온도", "비", "눈", "바람", "미세먼지",
        "weather", "temperature", "지진", "earthquake",
        "특보", "주의보", "경보", "태풍", "폭염", "한파",
        "검색", "뉴스", "최근", "현재", "지금", "오늘"
    ]
    for kw in tool_keywords:
        if kw in user_input_lower:
            return "tools"

    # RAG 검색이 필요한 키워드 (매뉴얼, 규정, 절차)
    rag_keywords = [
        "매뉴얼", "규정", "지침", "절차", "가이드", "방법",
        "행동요령", "대응", "조치", "기준", "정의",
        "어떻게", "무엇", "뭐야", "알려줘"
    ]
    for kw in rag_keywords:
        if kw in user_input_lower:
            return "retrieve"

    # 기본: 일반 대화로 종료
    return "end"


def agent_node(state: GraphState) -> Dict[str, Any]:
    """
    Agent 노드: 사용자 입력을 분석하고 다음 행동을 결정

    1. 사용자 입력 분석
    2. 컨텍스트 준비 (업로드된 파일, 히스토리 등)
    3. LLM 호출하여 의도 파악 또는 직접 응답
    4. next_action 결정
    """
    logger.info("🤖 [AGENT NODE] 실행 시작...")

    user_input = state.get("user_input", "")
    uploaded_context = state.get("uploaded_context", "")
    messages = state.get("messages", [])
    agent_mode = state.get("agent_mode", True)
    reasoning_mode = state.get("reasoning_mode", True)

    # 의도 파악
    detected_action = _detect_intent_with_llm(user_input)
    logger.info(f"🎯 [AGENT NODE] 감지된 의도: {detected_action}")

    # 보고서 요청이고 업로드된 컨텍스트가 있으면 report로
    if detected_action == "report" and uploaded_context:
        logger.info("📝 [AGENT NODE] 보고서 생성 경로로 라우팅")
        return {
            "next_action": "report",
            "messages": messages + [HumanMessage(content=user_input)]
        }

    # Agent 모드가 활성화되어 있고 tools가 필요하면
    if agent_mode and detected_action == "tools":
        logger.info("🔧 [AGENT NODE] 도구 사용 경로로 라우팅")
        return {
            "next_action": "tools",
            "messages": messages + [HumanMessage(content=user_input)]
        }

    # RAG 검색이 필요하면
    if state.get("enable_rag", True) and detected_action == "retrieve":
        logger.info("📚 [AGENT NODE] RAG 검색 경로로 라우팅")
        return {
            "next_action": "retrieve",
            "messages": messages + [HumanMessage(content=user_input)]
        }

    # 일반 대화: LLM으로 직접 응답 생성
    logger.info("💬 [AGENT NODE] 일반 대화 응답 생성")

    try:
        client = _get_llm_client()

        # 메시지 준비
        history = _convert_messages_to_history(messages)

        llm_messages = [
            {"role": "system", "content": _get_system_prompt()}
        ]

        # 히스토리 추가 (최근 3개만)
        for human_msg, ai_msg in history[-3:]:
            llm_messages.append({"role": "user", "content": human_msg})
            if ai_msg:
                llm_messages.append({"role": "assistant", "content": ai_msg})

        # 현재 질문 추가
        llm_messages.append({"role": "user", "content": user_input})

        # LLM 호출
        response = client.generate_response(
            messages=llm_messages,
            enable_thinking=reasoning_mode,
            temperature=0.6 if reasoning_mode else 0.4,
            stream=False
        )

        if response and response.choices:
            content = response.choices[0].message.content or ""
            # thinking 태그 제거
            final_response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        else:
            final_response = "죄송합니다. 응답을 생성하지 못했습니다."

        return {
            "next_action": "end",
            "final_response": final_response,
            "messages": messages + [
                HumanMessage(content=user_input),
                AIMessage(content=final_response)
            ]
        }

    except Exception as e:
        logger.error(f"❌ [AGENT NODE] LLM 호출 실패: {e}")
        return {
            "next_action": "end",
            "final_response": f"오류가 발생했습니다: {str(e)}",
            "messages": messages + [HumanMessage(content=user_input)]
        }


def tools_node(state: GraphState) -> Dict[str, Any]:
    """
    Tools 노드: 외부 API 도구 실행

    ExaoneAgentManager의 run_agent를 사용하여 도구 호출
    결과를 tool_results에 저장하고 Agent로 돌아감
    """
    logger.info("🔧 [TOOLS NODE] 실행 시작...")

    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    reasoning_mode = state.get("reasoning_mode", True)

    try:
        agent = _get_agent_manager()
        history = _convert_messages_to_history(messages[:-1])  # 마지막 메시지 제외

        # Agent 실행 (도구 호출 포함)
        result = agent.run_agent(user_input, history, reasoning_mode)

        if result.get("success"):
            tool_content = result.get("content", "")
            tool_calls = result.get("tool_calls", [])

            # 도구 결과 포맷팅
            tool_results_str = ""
            if tool_calls:
                tool_results_str = "\n\n**[도구 실행 결과]**\n"
                for tc in tool_calls:
                    tool_results_str += f"- {tc.get('name', 'unknown')}: 실행 완료\n"

            final_response = tool_content

            logger.info(f"✅ [TOOLS NODE] 도구 실행 완료: {len(tool_calls)}개 도구 호출")

            return {
                "tool_results": tool_results_str,
                "final_response": final_response,
                "next_action": "end",  # 도구 실행 후 종료
                "messages": messages + [AIMessage(content=final_response)]
            }
        else:
            error_msg = result.get("error", "알 수 없는 오류")
            logger.error(f"❌ [TOOLS NODE] 도구 실행 실패: {error_msg}")

            return {
                "tool_results": f"도구 실행 실패: {error_msg}",
                "next_action": "end",
                "final_response": f"도구 실행 중 오류가 발생했습니다: {error_msg}",
                "messages": messages
            }

    except Exception as e:
        logger.error(f"❌ [TOOLS NODE] 예외 발생: {e}")
        return {
            "tool_results": f"오류: {str(e)}",
            "next_action": "end",
            "final_response": f"도구 실행 중 오류가 발생했습니다: {str(e)}",
            "messages": messages
        }


def summarize_node(state: GraphState) -> Dict[str, Any]:
    """
    Summarize 노드: 도구 실행 결과가 너무 길 경우 LLM을 통해 요약
    """
    logger.info("🔍 [SUMMARIZE NODE] 실행 시작...")
    
    tool_results = state.get("tool_results", "")
    
    # 결과가 없거나 짧으면 패스 (예: 3000자 미만)
    if not tool_results or len(tool_results) < 3000:
        logger.info("✅ [SUMMARIZE NODE] 요약 불필요 (길이 적정)")
        return {} # 상태 변경 없음
        
    try:
        client = _get_llm_client()
        
        # 너무 긴 입력은 앞부분 20k만 사용하여 요약 시도
        input_text = tool_results[:20000]
        
        prompt = f"""
당신은 'Data Summarizer'입니다. 아래 도구(API) 실행 결과가 너무 길어 시스템이 처리하기 어렵습니다.
사용자의 질문에 답변하는 데 필요한 **핵심 정보만 추출하여 요약**하세요.
불필요한 중복 데이터, ID, 메타데이터 등은 제거하고, 중요한 수치나 텍스트 위주로 정리하세요.

[도구 실행 결과 (일부)]
{input_text}

[요약 결과]
"""
        messages = [{"role": "user", "content": prompt}]
        
        response = client.generate_response(
            messages=messages,
            enable_thinking=False, # 요약은 빠른 처리 우선
            temperature=0.2,
            max_tokens=2048
        )
        
        if response and response.choices:
            content = response.choices[0].message.content.strip()
            # thinking 태그 제거
            summary = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            logger.info(f"✅ [SUMMARIZE NODE] 요약 완료 ({len(tool_results)}자 -> {len(summary)}자)")
            return {
                "tool_results": f"**[AI 요약된 도구 결과]**\n{summary}"
            }
            
    except Exception as e:
        logger.error(f"❌ [SUMMARIZE NODE] 요약 실패: {e}")
        # 실패 시 안전하게 앞부분만 자름
        return {
            "tool_results": tool_results[:3000] + "\n...(요약 실패로 절삭됨)"
        }
    
    return {}


def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve 노드: RAG 검색 실행

    AdvancedRAGSystem의 search를 사용하여 문서 검색
    결과를 rag_results에 저장하고 Agent로 돌아가서 최종 응답 생성
    """
    logger.info("📚 [RETRIEVE NODE] 실행 시작...")

    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    selected_pdf = state.get("selected_pdf", None)
    reasoning_mode = state.get("reasoning_mode", True)

    try:
        rag = _get_rag_system()
        client = _get_llm_client()

        # RAG 검색 실행
        results = rag.search(user_input, selected_pdf, top_k=5)

        if results:
            # 검색 결과 포맷팅
            rag_context = rag.format_search_results(results, selected_pdf)

            logger.info(f"✅ [RETRIEVE NODE] 검색 완료: {len(results)}개 문서 발견")

            # 검색 결과를 바탕으로 응답 생성
            llm_messages = [
                {"role": "system", "content": _get_system_prompt()},
                {"role": "user", "content": f"""다음 검색된 문서 내용을 바탕으로 사용자 질문에 답변해주세요.

**검색된 문서:**
{rag_context}

**사용자 질문:**
{user_input}

답변 시 출처를 명시해주세요."""}
            ]

            response = client.generate_response(
                messages=llm_messages,
                enable_thinking=reasoning_mode,
                temperature=0.4,
                stream=False
            )

            if response and response.choices:
                content = response.choices[0].message.content or ""
                final_response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            else:
                final_response = f"검색 결과를 찾았으나 응답 생성에 실패했습니다.\n\n{rag_context}"

            return {
                "rag_results": rag_context,
                "reference_docs": rag_context,
                "final_response": final_response,
                "next_action": "end",
                "messages": messages + [AIMessage(content=final_response)]
            }
        else:
            logger.warning("⚠️ [RETRIEVE NODE] 관련 문서를 찾지 못함")
            return {
                "rag_results": "관련 문서를 찾지 못했습니다.",
                "next_action": "end",
                "final_response": "죄송합니다. 관련된 문서를 찾지 못했습니다. 다른 키워드로 검색해 보시겠어요?",
                "messages": messages
            }

    except Exception as e:
        logger.error(f"❌ [RETRIEVE NODE] 예외 발생: {e}")
        return {
            "rag_results": f"검색 오류: {str(e)}",
            "next_action": "end",
            "final_response": f"문서 검색 중 오류가 발생했습니다: {str(e)}",
            "messages": messages
        }


def report_node(state: GraphState) -> Dict[str, Any]:
    """
    Report 노드: 보고서 생성 파이프라인 실행

    ReportGenerator의 4단계 파이프라인 실행:
    1. parse_document: 문서 구조 분석
    2. plan_tools: 도구 사용 계획
    3. fill_report: 내용 채우기
    4. export_to_docx: 문서 내보내기 (선택적)

    결과를 report_output에 저장하고 END로 직접 이동
    """
    logger.info("📝 [REPORT NODE] 실행 시작...")

    user_input = state.get("user_input", "")
    uploaded_context = state.get("uploaded_context", "")
    messages = state.get("messages", [])

    try:
        generator = _get_report_generator()

        # 입력 텍스트 준비 (업로드된 컨텍스트 또는 사용자 입력)
        input_text = uploaded_context if uploaded_context else user_input

        if not input_text:
            return {
                "report_output": "보고서 생성을 위한 입력이 없습니다.",
                "final_response": "보고서를 생성하려면 문서를 업로드하거나 내용을 입력해주세요.",
                "messages": messages
            }

        # Phase 1: 문서 구조 분석
        logger.info("📄 [REPORT NODE] Phase 1: 문서 구조 분석...")
        structure = generator.parse_document(input_text)

        if structure.get("status") == "incomplete":
            # 추가 정보 필요
            clarification = structure.get("clarification_question", "추가 정보가 필요합니다.")
            return {
                "report_output": None,
                "final_response": f"보고서 작성을 위해 추가 정보가 필요합니다:\n{clarification}",
                "messages": messages + [AIMessage(content=clarification)]
            }

        if structure.get("status") == "error":
            return {
                "report_output": None,
                "final_response": "문서 구조 분석에 실패했습니다. 다시 시도해주세요.",
                "messages": messages
            }

        # Phase 2: 도구 계획
        logger.info("🧠 [REPORT NODE] Phase 2: 도구 계획 수립...")
        structure = generator.plan_tools(structure)

        # Phase 3 & 4: 내용 채우기 및 생성
        logger.info("🛠️ [REPORT NODE] Phase 3: 내용 작성...")
        structure = generator.fill_report(structure)

        # 최종 보고서 가져오기
        report_md = structure.get("full_report_md", "보고서 생성에 실패했습니다.")

        # DOCX 내보내기 (선택적)
        docx_path = ""
        try:
            docx_path = generator.export_to_docx(structure)
            if docx_path:
                report_md += f"\n\n📁 **문서 저장됨:** `{docx_path}`"
        except Exception as e:
            logger.warning(f"DOCX 내보내기 실패: {e}")

        logger.info("✅ [REPORT NODE] 보고서 생성 완료")

        return {
            "report_output": report_md,
            "final_response": report_md,
            "messages": messages + [AIMessage(content=report_md)]
        }

    except Exception as e:
        logger.error(f"❌ [REPORT NODE] 예외 발생: {e}")
        return {
            "report_output": f"보고서 생성 오류: {str(e)}",
            "final_response": f"보고서 생성 중 오류가 발생했습니다: {str(e)}",
            "messages": messages
        }


def _get_system_prompt() -> str:
    """시스템 프롬프트 반환"""
    from config import SYSTEM_MESSAGE
    return SYSTEM_MESSAGE
