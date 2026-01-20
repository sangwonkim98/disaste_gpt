"""
Query Pipeline 노드 구현

1. analyzer_node: 사용자 입력 분석 → 실행 계획 수립
2. tool_executor_node: API 도구 실행
3. rag_executor_node: RAG 검색 실행
4. pdf_executor_node: PDF 분석 실행
5. synthesizer_node: 결과 종합 → 최종 응답 생성
"""

import logging
import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage

from config import VLLM_SERVER_URL, VLLM_API_KEY, LLM_MODEL_NAME, SYSTEM_MESSAGE
from graph.query_state import (
    QueryState, ExecutionPlan, ExecutionResult,
    TOOL_METADATA, TOOL_DEPENDENCIES, KEYWORD_TOOL_MAP, RAG_KEYWORDS
)

logger = logging.getLogger(__name__)

# ===== 지연 로딩 싱글톤 =====
_llm_client = None
_agent_tools = None
_rag_system = None


def _get_llm_client():
    global _llm_client
    if _llm_client is None:
        from core.llm_client import ExaoneClient
        _llm_client = ExaoneClient(
            server_url=VLLM_SERVER_URL,
            api_key=VLLM_API_KEY,
            model_name=LLM_MODEL_NAME
        )
    return _llm_client


def _get_agent_tools():
    global _agent_tools
    if _agent_tools is None:
        from services.agent_tools import exaone_agent_tools
        _agent_tools = exaone_agent_tools
    return _agent_tools


def _get_rag_system():
    """RAG 시스템 싱글톤 반환 (PDF 인덱스 자동 빌드)"""
    global _rag_system
    if _rag_system is None:
        from services.rag_engine import AdvancedRAGSystem
        from config import PDF_FILES
        from pathlib import Path

        _rag_system = AdvancedRAGSystem()

        # PDF 파일이 있으면 인덱스 빌드
        existing_pdfs = [pdf for pdf in PDF_FILES if Path(pdf).exists()]
        if existing_pdfs:
            logger.info(f"📚 RAG 인덱스 빌드 시작: {len(existing_pdfs)}개 PDF")
            _rag_system.build_index(existing_pdfs)
            logger.info(f"✅ RAG 인덱스 빌드 완료")
        else:
            logger.warning("⚠️ PDF 파일 없음 - RAG 검색이 비활성화됩니다")

    return _rag_system


def _debug_log(state: QueryState, message: str, data: Any = None):
    """디버그 모드일 때만 상세 로깅"""
    if state.get("debug_mode", False):
        if data:
            logger.info(f"🔍 [DEBUG] {message}: {json.dumps(data, ensure_ascii=False, default=str)[:500]}")
        else:
            logger.info(f"🔍 [DEBUG] {message}")


# =============================================================================
# 1. ANALYZER NODE
# =============================================================================

ANALYZER_SYSTEM_PROMPT = """당신은 재난대응 AI 시스템의 'Query Analyzer'입니다.
사용자의 질문을 분석하여 어떤 정보 소스가 필요한지 판단하세요.

## 사용 가능한 정보 소스

### 1. 외부 API 도구 (tools)
실시간 데이터가 필요할 때 사용:
- serpapi_web_search: 최신 뉴스, 사건 검색
- kma_get_ultra_srt_ncst: 현재 실시간 날씨
- kma_get_vilage_fcst: 단기예보 (3일)
- kma_get_mid_land_fcst: 중기예보 (3~10일)
- kma_get_mid_ta: 중기기온예보
- kma_get_wthr_wrn_msg: 기상특보 (태풍, 호우 등)
- kma_get_pwn_status: 예비특보
- kma_get_eqk_msg_list: 지진 목록
- kma_get_eqk_msg: 지진 상세

### 2. 매뉴얼 검색 (rag)
공무원 대응 절차, 행동요령이 필요할 때 사용:
- 풍수해 대응 매뉴얼
- 재난 현장조치 행동매뉴얼
- 위기관리 표준매뉴얼

### 3. 사용자 PDF (pdf)
사용자가 업로드한 문서 분석이 필요할 때 사용

## 출력 형식 (JSON)
```json
{
  "need_tools": true/false,
  "need_rag": true/false,
  "need_pdf": true/false,
  "tool_list": ["tool_name1", "tool_name2"],
  "tool_params": {
    "tool_name1": {"location": "서울"},
    "tool_name2": {}
  },
  "tool_reasoning": "도구 선택 이유",
  "rag_query": "매뉴얼 검색 쿼리 (재구성)",
  "rag_reasoning": "RAG 필요 이유",
  "pdf_task": "extract_info | use_as_template | null",
  "confidence": 0.9
}
```

## 중요 규칙
1. 실시간 정보(날씨, 뉴스, 지진)가 필요하면 반드시 tools 사용
2. 대응 절차, 행동요령 질문은 rag 사용
3. 복합 질문은 tools + rag 모두 true
4. 단순 인사나 일반 대화는 모두 false

오직 JSON만 출력하세요."""


def _extract_location(user_input: str) -> Optional[str]:
    """사용자 입력에서 지역명 추출"""
    # 자주 쓰이는 지역명 목록
    locations = [
        "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
        "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
        "수원", "성남", "용인", "고양", "창원", "청주", "천안", "전주",
        "포항", "김해", "안산", "안양", "남양주", "화성", "평택"
    ]
    for loc in locations:
        if loc in user_input:
            return loc
    return None


def _rule_based_analysis(user_input: str, has_pdf: bool) -> ExecutionPlan:
    """규칙 기반 분석 (LLM 실패 시 fallback)"""
    user_lower = user_input.lower()

    need_tools = False
    need_rag = False
    need_pdf = has_pdf
    tool_list = []
    tool_params = {}

    location = _extract_location(user_input)

    # 키워드 매칭으로 도구 선택
    for keyword, tools in KEYWORD_TOOL_MAP.items():
        if keyword in user_lower:
            need_tools = True
            for tool in tools:
                if tool not in tool_list:
                    tool_list.append(tool)
                    if TOOL_METADATA.get(tool, {}).get("requires_location") and location:
                        tool_params[tool] = {"location": location}
                    else:
                        tool_params[tool] = {}

    # RAG 키워드 매칭
    for keyword in RAG_KEYWORDS:
        if keyword in user_lower:
            need_rag = True
            break

    # 도구 의존성 체이닝
    for tool in list(tool_list):
        if tool in TOOL_DEPENDENCIES:
            dep = TOOL_DEPENDENCIES[tool]
            if dep.get("auto_chain"):
                for suggested in dep.get("suggests", []):
                    if suggested not in tool_list:
                        tool_list.append(suggested)
                        if TOOL_METADATA.get(suggested, {}).get("requires_location") and location:
                            tool_params[suggested] = {"location": location}
                        else:
                            tool_params[suggested] = {}

    return ExecutionPlan(
        need_tools=need_tools,
        need_rag=need_rag,
        need_pdf=need_pdf,
        tool_list=tool_list,
        tool_params=tool_params,
        tool_reasoning="규칙 기반 분석 (키워드 매칭)",
        rag_query=user_input if need_rag else None,
        rag_reasoning="대응 절차/매뉴얼 관련 키워드 감지" if need_rag else None,
        pdf_task="extract_info" if need_pdf else None,
        confidence=0.7
    )


def analyzer_node(state: QueryState) -> Dict[str, Any]:
    """
    사용자 입력을 분석하여 실행 계획을 수립합니다.

    1. LLM으로 의도 분석 시도
    2. 실패 시 규칙 기반 분석으로 fallback
    3. 도구 의존성 체이닝 적용
    """
    logger.info("🔎 [ANALYZER] 실행 시작...")

    user_input = state.get("user_input", "")
    has_pdf = bool(state.get("uploaded_pdf_content"))
    debug_mode = state.get("debug_mode", False)

    _debug_log(state, "입력", {"user_input": user_input, "has_pdf": has_pdf})

    # 빈 입력 처리
    if not user_input.strip():
        return {
            "execution_plan": None,
            "next_node": "end",
            "final_response": "질문을 입력해주세요."
        }

    # LLM 기반 분석 시도
    try:
        client = _get_llm_client()

        # 컨텍스트 구성
        context = f"사용자 질문: {user_input}"
        if has_pdf:
            context += "\n[사용자가 PDF 파일을 업로드함]"

        messages = [
            {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
            {"role": "user", "content": context}
        ]

        response = client.generate_response(
            messages=messages,
            temperature=0.0,
            enable_thinking=False,
            stream=False
        )

        if response and response.choices:
            content = response.choices[0].message.content.strip()

            # JSON 추출
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # ExecutionPlan 구성
            plan = ExecutionPlan(
                need_tools=result.get("need_tools", False),
                need_rag=result.get("need_rag", False),
                need_pdf=result.get("need_pdf", has_pdf),
                tool_list=result.get("tool_list", []),
                tool_params=result.get("tool_params", {}),
                tool_reasoning=result.get("tool_reasoning", ""),
                rag_query=result.get("rag_query"),
                rag_reasoning=result.get("rag_reasoning"),
                pdf_task=result.get("pdf_task"),
                confidence=result.get("confidence", 0.8)
            )

            # 지역 파라미터 보완
            location = _extract_location(user_input)
            if location:
                for tool in plan["tool_list"]:
                    if TOOL_METADATA.get(tool, {}).get("requires_location"):
                        if tool not in plan["tool_params"]:
                            plan["tool_params"][tool] = {}
                        if "location" not in plan["tool_params"][tool]:
                            plan["tool_params"][tool]["location"] = location

            # 도구 의존성 체이닝
            for tool in list(plan["tool_list"]):
                if tool in TOOL_DEPENDENCIES:
                    dep = TOOL_DEPENDENCIES[tool]
                    if dep.get("auto_chain"):
                        for suggested in dep.get("suggests", []):
                            if suggested not in plan["tool_list"]:
                                plan["tool_list"].append(suggested)
                                if TOOL_METADATA.get(suggested, {}).get("requires_location") and location:
                                    plan["tool_params"][suggested] = {"location": location}

            logger.info(f"✅ [ANALYZER] LLM 분석 완료: tools={plan['need_tools']}, rag={plan['need_rag']}, pdf={plan['need_pdf']}")
            _debug_log(state, "실행 계획", plan)

            # 다음 노드 결정
            if plan["need_tools"] or plan["need_rag"] or plan["need_pdf"]:
                next_node = "executor"
            else:
                next_node = "direct_response"

            return {
                "execution_plan": plan,
                "next_node": next_node
            }

    except Exception as e:
        logger.warning(f"⚠️ [ANALYZER] LLM 분석 실패, 규칙 기반으로 전환: {e}")

    # Fallback: 규칙 기반 분석
    plan = _rule_based_analysis(user_input, has_pdf)
    logger.info(f"📋 [ANALYZER] 규칙 기반 분석 완료: tools={plan['need_tools']}, rag={plan['need_rag']}")

    if plan["need_tools"] or plan["need_rag"] or plan["need_pdf"]:
        next_node = "executor"
    else:
        next_node = "direct_response"

    return {
        "execution_plan": plan,
        "next_node": next_node
    }


# =============================================================================
# 2. TOOL EXECUTOR NODE
# =============================================================================

def tool_executor_node(state: QueryState) -> Dict[str, Any]:
    """
    실행 계획에 따라 API 도구들을 실행합니다.
    """
    logger.info("🔧 [TOOL_EXECUTOR] 실행 시작...")

    plan = state.get("execution_plan")
    if not plan or not plan.get("need_tools"):
        return {}

    tool_list = plan.get("tool_list", [])
    tool_params = plan.get("tool_params", {})

    _debug_log(state, "실행할 도구", {"tools": tool_list, "params": tool_params})

    agent_tools = _get_agent_tools()
    results = {}

    for tool_name in tool_list:
        try:
            params = tool_params.get(tool_name, {})
            logger.info(f"  🛠️ 실행: {tool_name}({params})")

            result = agent_tools.execute_tool(tool_name, params)
            results[tool_name] = json.loads(result) if isinstance(result, str) else result

            logger.info(f"  ✅ 완료: {tool_name}")

        except Exception as e:
            logger.error(f"  ❌ 실패: {tool_name} - {e}")
            results[tool_name] = {"error": str(e)}

    # 현재 execution_result 가져오기 (없으면 초기화)
    current_result = state.get("execution_result") or {}
    current_result["tool_results"] = results

    logger.info(f"✅ [TOOL_EXECUTOR] {len(results)}개 도구 실행 완료")

    return {
        "execution_result": current_result
    }


# =============================================================================
# 3. RAG EXECUTOR NODE
# =============================================================================

def rag_executor_node(state: QueryState) -> Dict[str, Any]:
    """
    매뉴얼 RAG 검색을 실행합니다.
    """
    logger.info("📚 [RAG_EXECUTOR] 실행 시작...")

    plan = state.get("execution_plan")
    if not plan or not plan.get("need_rag"):
        return {}

    user_input = state.get("user_input", "")
    rag_query = plan.get("rag_query") or user_input
    selected_manual = state.get("selected_manual")

    _debug_log(state, "RAG 검색", {"query": rag_query, "manual": selected_manual})

    try:
        rag = _get_rag_system()
        results = rag.search(rag_query, selected_manual, top_k=5)

        # 결과 포맷팅
        formatted_results = []
        if results:
            for i, doc in enumerate(results):
                formatted_results.append({
                    "rank": i + 1,
                    "content": doc.page_content[:500] if hasattr(doc, 'page_content') else str(doc)[:500],
                    "source": doc.metadata.get("source", "unknown") if hasattr(doc, 'metadata') else "unknown"
                })

        logger.info(f"✅ [RAG_EXECUTOR] {len(formatted_results)}개 문서 검색 완료")

        # 현재 execution_result 가져오기
        current_result = state.get("execution_result") or {}
        current_result["rag_results"] = formatted_results

        return {
            "execution_result": current_result
        }

    except Exception as e:
        logger.error(f"❌ [RAG_EXECUTOR] 검색 실패: {e}")
        current_result = state.get("execution_result") or {}
        current_result["rag_results"] = [{"error": str(e)}]
        return {
            "execution_result": current_result
        }


# =============================================================================
# 4. PDF EXECUTOR NODE
# =============================================================================

def pdf_executor_node(state: QueryState) -> Dict[str, Any]:
    """
    사용자 업로드 PDF를 분석합니다.
    """
    logger.info("📄 [PDF_EXECUTOR] 실행 시작...")

    plan = state.get("execution_plan")
    if not plan or not plan.get("need_pdf"):
        return {}

    pdf_content = state.get("uploaded_pdf_content")
    if not pdf_content:
        return {}

    pdf_task = plan.get("pdf_task", "extract_info")

    _debug_log(state, "PDF 분석", {"task": pdf_task, "content_len": len(pdf_content)})

    # 현재는 단순히 내용 전달 (추후 고도화)
    current_result = state.get("execution_result") or {}
    current_result["pdf_results"] = {
        "task": pdf_task,
        "content_preview": pdf_content[:1000] if pdf_content else "",
        "full_content": pdf_content
    }

    logger.info(f"✅ [PDF_EXECUTOR] PDF 분석 완료 (task: {pdf_task})")

    return {
        "execution_result": current_result
    }


# =============================================================================
# 5. SYNTHESIZER NODE
# =============================================================================

SYNTHESIZER_SYSTEM_PROMPT = """당신은 재난대응 AI 어시스턴트입니다.
수집된 정보를 바탕으로 사용자 질문에 명확하고 유용하게 답변하세요.

## 답변 원칙
1. 정확성: 제공된 데이터를 기반으로 사실만 전달
2. 명확성: 핵심 정보를 먼저, 부가 정보는 뒤에
3. 실용성: 공무원이 바로 활용할 수 있는 형태로
4. 출처 명시: 정보의 출처를 명확히 (API 결과, 매뉴얼 등)

## 포맷
- 날씨 정보: 기온, 강수, 바람 등 핵심 수치 포함
- 특보 정보: 발효 중인 특보와 주의사항
- 매뉴얼 정보: 단계별 조치사항, 출처 페이지
- 복합 정보: 현황 → 대응방안 순서로 정리"""


def synthesizer_node(state: QueryState) -> Dict[str, Any]:
    """
    수집된 모든 결과를 종합하여 최종 응답을 생성합니다.
    """
    logger.info("📝 [SYNTHESIZER] 실행 시작...")

    user_input = state.get("user_input", "")
    plan = state.get("execution_plan")
    result = state.get("execution_result") or {}
    messages = state.get("messages", [])
    reasoning_mode = state.get("reasoning_mode", True)

    tool_results = result.get("tool_results", {})
    rag_results = result.get("rag_results", [])
    pdf_results = result.get("pdf_results")

    _debug_log(state, "종합할 결과", {
        "tool_count": len(tool_results),
        "rag_count": len(rag_results),
        "has_pdf": bool(pdf_results)
    })

    # 컨텍스트 구성
    context_parts = []

    # 도구 실행 결과
    if tool_results:
        context_parts.append("## 📊 실시간 데이터 (API 조회 결과)")
        for tool_name, data in tool_results.items():
            tool_desc = TOOL_METADATA.get(tool_name, {}).get("description", tool_name)
            context_parts.append(f"\n### {tool_desc}")
            context_parts.append(f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)[:2000]}\n```")

    # RAG 검색 결과
    if rag_results:
        context_parts.append("\n## 📚 매뉴얼 검색 결과")
        for doc in rag_results[:3]:  # 상위 3개만
            if "error" not in doc:
                context_parts.append(f"\n**출처:** {doc.get('source', 'unknown')}")
                context_parts.append(f"{doc.get('content', '')}")

    # PDF 분석 결과
    if pdf_results:
        context_parts.append("\n## 📄 업로드된 PDF 분석")
        context_parts.append(f"**분석 유형:** {pdf_results.get('task', 'extract_info')}")
        context_parts.append(f"{pdf_results.get('content_preview', '')[:500]}")

    context = "\n".join(context_parts)

    # LLM으로 최종 응답 생성
    try:
        client = _get_llm_client()

        llm_messages = [
            {"role": "system", "content": SYNTHESIZER_SYSTEM_PROMPT},
            {"role": "user", "content": f"""다음 정보를 바탕으로 사용자 질문에 답변해주세요.

**사용자 질문:** {user_input}

**수집된 정보:**
{context}

답변:"""}
        ]

        response = client.generate_response(
            messages=llm_messages,
            temperature=0.4,
            enable_thinking=reasoning_mode,
            stream=False
        )

        if response and response.choices:
            content = response.choices[0].message.content or ""
            # thinking 태그 제거
            final_response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        else:
            final_response = f"정보 조회 결과:\n\n{context}"

        logger.info("✅ [SYNTHESIZER] 응답 생성 완료")

        return {
            "final_response": final_response,
            "next_node": "end",
            "messages": messages + [
                HumanMessage(content=user_input),
                AIMessage(content=final_response)
            ]
        }

    except Exception as e:
        logger.error(f"❌ [SYNTHESIZER] 응답 생성 실패: {e}")
        # Fallback: 원본 데이터 반환
        return {
            "final_response": f"정보 조회 결과:\n\n{context}\n\n(응답 생성 중 오류 발생: {e})",
            "next_node": "end",
            "messages": messages + [HumanMessage(content=user_input)]
        }


# =============================================================================
# 6. DIRECT RESPONSE NODE (도구 없이 직접 응답)
# =============================================================================

def direct_response_node(state: QueryState) -> Dict[str, Any]:
    """
    도구 호출 없이 LLM으로 직접 응답합니다.
    (일반 대화, 인사 등)
    """
    logger.info("💬 [DIRECT_RESPONSE] 실행 시작...")

    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    reasoning_mode = state.get("reasoning_mode", True)

    try:
        client = _get_llm_client()

        llm_messages = [
            {"role": "system", "content": SYSTEM_MESSAGE}
        ]

        # 최근 대화 히스토리 추가
        for msg in messages[-6:]:
            if isinstance(msg, HumanMessage):
                llm_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                llm_messages.append({"role": "assistant", "content": msg.content})

        llm_messages.append({"role": "user", "content": user_input})

        response = client.generate_response(
            messages=llm_messages,
            temperature=0.6,
            enable_thinking=reasoning_mode,
            stream=False
        )

        if response and response.choices:
            content = response.choices[0].message.content or ""
            final_response = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        else:
            final_response = "죄송합니다. 응답을 생성하지 못했습니다."

        logger.info("✅ [DIRECT_RESPONSE] 응답 생성 완료")

        return {
            "final_response": final_response,
            "next_node": "end",
            "messages": messages + [
                HumanMessage(content=user_input),
                AIMessage(content=final_response)
            ]
        }

    except Exception as e:
        logger.error(f"❌ [DIRECT_RESPONSE] 응답 생성 실패: {e}")
        return {
            "final_response": f"오류가 발생했습니다: {str(e)}",
            "next_node": "end",
            "messages": messages + [HumanMessage(content=user_input)]
        }
