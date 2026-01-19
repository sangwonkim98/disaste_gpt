"""
[Chat Manager]
대화 흐름 제어 및 통합 관리 (Broker)
사용자 메시지 -> [Agent/RAG/Report] 분기 처리 -> LLM 호출 -> 응답 스트리밍
"""

import logging
import json
import re
from typing import List, Dict, Generator, Tuple, Optional
from pathlib import Path
import requests
from openai import OpenAI

from config import (
    VLLM_SERVER_URL, VLLM_API_KEY, LLM_MODEL_NAME, ENABLE_REASONING,
    MAX_TOKENS, TEMPERATURE, TOP_P, TOP_K, MIN_P, SYSTEM_MESSAGE, TOP_K_RESULTS
)
from services.rag_engine import AdvancedRAGSystem
from core.agent_manager import exaone_agent

logger = logging.getLogger(__name__)

class ExaoneClient:
    """EXAONE 모델 (VLLM 서버) 통신 클라이언트"""
    def __init__(self, server_url, api_key, model_name="LGAI-EXAONE/EXAONE-4.0-32B-AWQ"):
        self.server_url = server_url
        self.api_key = api_key
        self.model_name = model_name
        
        # OpenAI SDK 호환 클라이언트 사용
        self.client = OpenAI(api_key=api_key, base_url=server_url, timeout=180.0)

    def generate_response(self, messages: List[Dict], enable_thinking: bool = False,
                          temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
                          top_p: float = TOP_P, stream: bool = True):
        """
        LLM 생성 요청 전송
        - extra_body를 통해 'enable_thinking' 파라미터 전달 (Reasoning 모드 제어)
        """
        try:
            response = self.client.chat.completions.create(
                model = self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )
            return response
        except Exception as e:
            logger.error(f"EXAONE API 호출 실패: {e}")
            return None


class ChatManager:
    """
    [핵심 클래스] 대화 세션 관리자
    - 역할: 사용자 입력 수신, 적절한 처리기(Agent, RAG) 호출, 최종 프롬프트 조립, 응답 생성
    """

    def __init__(self):
        self.rag_system = AdvancedRAGSystem()
        self.exaone_client = ExaoneClient(
            server_url=VLLM_SERVER_URL,
            api_key=VLLM_API_KEY,
            model_name = LLM_MODEL_NAME,
        )
        # 추론 모드 여부에 따른 파라미터 설정
        self.reasoning_temperature = 0.6 # 창의적 사고 필요
        self.non_reasoning_temperature = 0.4 # 정확한 정보 전달 중요

    def initialize_system(self, pdf_paths: List[str]):
        """시스템 시작 시 RAG 인덱스 빌드"""
        logger.info("시스템 초기화 시작...")
        self.rag_system.build_index(pdf_paths)
        logger.info("시스템 초기화 완료")
    
    def _prepare_messages(self, user_message: str, history: List[List[str]], rag_context: str = None, agent_context: str = None, uploaded_context: str = None):
        """
        [Prompt Engineering] LLM에 보낼 최종 메시지 컨텍스트 조립
        - System Prompt + 시간 정보 + 업로드 파일 내용 + Agent 결과 + RAG 검색 결과 + 대화 히스토리 + 현재 질문
        """
        messages = []

        # 1. 시스템 프롬프트 및 시간 정보 주입
        from datetime import datetime
        now_str = datetime.now().strftime("%Y년 %m월 %d일 %A %H시 %M분")
        system_content = f"현재 시각: {now_str}\n\n{SYSTEM_MESSAGE}"

        # 2. [파일 내용] 업로드된 문서 내용을 별도 섹션으로 주입
        if uploaded_context:
            truncated_context = uploaded_context[:15000] # 토큰 제한 고려하여 길이 제한
            if len(uploaded_context) > 15000:
                truncated_context += "\n...(내용이 너무 길어 생략됨)..."
            system_content += f"\n\n=== [사용자 업로드 파일 내용 (참고 자료)] ===\n{truncated_context}\n\n위 파일 내용을 분석하거나 참고하여 답변하십시오."

        # 3. [Agent 결과] 툴 실행 결과 주입
        if agent_context:
            system_content += f"\n\n === Agent 실행 결과 ===\n{agent_context}\n\n위 결과를 참고하여 정확하고 유용한 답변을 제공하세요."
        
        # 4. [RAG 결과] 검색된 규정 문서 주입
        if rag_context:
            system_content += f"\n\n === 참고 문서 ====\n{rag_context}\n\n위 문서 내용을 참고하여 정확하고 유용한 답변을 제공하세요."
        
        messages.append({"role": "system", "content": system_content})

        # 5. 대화 히스토리 (최근 10턴 유지)
        max_history_turns = 10
        if len(history) > max_history_turns:
            history = history[-max_history_turns:]

        for human_msg, ai_msg in history:
            if human_msg: messages.append({"role": "user", "content": human_msg.strip()})
            if ai_msg: messages.append({"role": "assistant", "content": self._clean_ai_message(ai_msg.strip())})

        messages.append({"role": "user", "content": user_message})
        
        # [DEBUG] 프롬프트 조립 결과 파일 저장 (PDF 내용 확인용)
        try:
            debug_path = Path("hs_code/debug_prompt.log")
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
            print(f"🐛 [DEBUG] 조립된 전체 프롬프트가 '{debug_path}'에 저장되었습니다.")
        except Exception as e:
            print(f"❌ [DEBUG] 프롬프트 저장 실패: {e}")

        return messages

    def _clean_ai_message(self, ai_msg: str) -> str:
        """히스토리 오염 방지: 이전 응답에서 로그/메타데이터 제거"""
        # ... (정규식 제거 로직 생략) ...
        return ai_msg # 실제로는 정제 로직 적용됨

    def process_message(self, user_message: str, history: List[List[str]], agent_mode: bool = False, 
                    reasoning_mode: bool = True, enable_reasoning: bool = True, enable_rag: bool = True,
                    selected_pdf_path: Optional[str] = None, reset_state: bool = False, uploaded_context: str = "") -> Generator[Tuple[List[List[str]], str], None, None]:
        """
        [Main Loop] 메시지 처리 메인 파이프라인
        """
        
        # [TRACE] 시작
        print(f"\n{'='*60}")
        print(f"🚀 [TRACE] 사용자 입력 수신: \"{user_message}\"")
        print(f"   - 설정: Agent={agent_mode}, RAG={enable_rag}, Reasoning={reasoning_mode}")
        print(f"{'='*60}")

        if reset_state: self._reset_processing_state()
        
        # 1. [Special Flow] 보고서 생성 요청인지 확인
        if "보고서 생성" in user_message or "보고서 작성" in user_message:
            print("🚦 [TRACE] 라우팅 결정: >> [보고서 생성 트랙] (ReportGenerator)")
            logger.info("📄 [PROCESS] 보고서 생성 요청 감지")
            
            # ... (기존 보고서 생성 로직) ...
            # 여기서는 yield 부분만 간략히 유지하고 실제 로직은 generator 호출부로 가정
            # (실제 코드가 길어서 문맥 유지를 위해 주석 처리된 부분은 그대로 두거나 생략)
            new_history = history + [[user_message, "📄 보고서 생성을 시작합니다..."]]
            yield (new_history, "")
            
            try:
                from core.generator import ReportGenerator
                generator = ReportGenerator()
                target_text = uploaded_context if uploaded_context else user_message
                
                print("   Step 1: 문서 구조 분석 (parse_document)")
                new_history[-1][1] = "📄 [1단계] 문서 구조 분석 중..."
                yield (new_history, "")
                structure = generator.parse_document(target_text)
                
                if structure.get("status") == "incomplete":
                    print(f"   ⚠️ 정보 부족: {structure.get('missing_fields')}")
                    new_history[-1][1] = f"🤔 **확인 필요:** {structure.get('clarification_question')}"
                    yield (new_history, "")
                    return

                if not structure or not structure.get("sections"):
                    print("   ❌ 구조 분석 실패")
                    new_history[-1][1] = "❌ 문서 구조를 파악하지 못했습니다."
                    yield (new_history, "")
                    return

                print("   Step 2: 툴 플래닝 (plan_tools)")
                new_history[-1][1] = f"🛠️ [2단계] 도구 사용 계획 수립 중..."
                yield (new_history, "")
                structure = generator.plan_tools(structure)

                print("   Step 3: 실행 및 작성 (fill_report)")
                new_history[-1][1] = f"⚡ [3단계] 데이터 수집 및 보고서 작성 중..."
                yield (new_history, "")
                structure = generator.fill_report(structure)
                
                print("   Step 4: DOCX 내보내기")
                docx_path = generator.export_to_docx(structure)
                
                final_report = structure.get("full_report_md", "")
                if docx_path:
                    final_report += f"\n\n---\n### 💾 [보고서 다운로드]\nDOCX 파일이 생성되었습니다: `{Path(docx_path).name}`"
                
                print("🏁 [TRACE] 보고서 생성 완료")
                new_history[-1][1] = final_report
                yield (new_history, "")
                return

            except Exception as e:
                print(f"❌ [TRACE] 보고서 생성 중 에러: {e}")
                logger.error(f"보고서 생성 실패: {e}")
                new_history[-1][1] = f"❌ 보고서 생성 중 오류가 발생했습니다: {str(e)}"
                yield (new_history, "")
                return
        
        # 기본 변수 초기화
        rag_context = ""
        rag_evidence_text = ""  # [FIX] RAG 근거 텍스트 초기화
        agent_context = None
        new_history = history + [[user_message, ""]]
        yield (new_history, "")

        # 2. [Agent Mode] 에이전트 실행 (툴 사용이 필요한 경우)
        if agent_mode and exaone_agent.is_available():
            print("🚦 [TRACE] 라우팅 결정: >> [Agent 트랙] (exaone_agent)")
            logger.info("🤖 [PROCESS] Agent 모드 활성화")
            new_history[-1][1] = "🤖 EXAONE Agent 실행 시작...\n\n"
            yield (new_history, "")

            # Agent 상태 콜백 (UI 로그 표시용)
            def status_callback(api_name: str, message: str):
                pass 

            exaone_agent.set_status_callback(status_callback)
            agent_results = {"success": False}
            
            try:
                cleaned_history = []
                for h, a in (history or []):
                    if a: cleaned_history.append([h, self._clean_ai_message(a)])
                
                # Agent 실행 및 스트리밍 응답 처리
                for chunk in exaone_agent.run_agent_stream(user_message, cleaned_history, True):
                    ctype = chunk.get("type")
                    # [TRACE] 툴 실행 로그
                    if ctype == "tool_executing":
                        print(f"   🛠️ [TOOL] 실행: {chunk.get('tool_name')}")
                    elif ctype == "tool_complete":
                        print(f"   ✅ [TOOL] 완료: {chunk.get('tool_name')}")
                    
                    # UI 스트리밍 로직 (Clean & Simple: 채팅창에는 답변만 표시)
                    if ctype == "content":
                        new_history[-1][1] += chunk.get("content", "")
                        yield (new_history, "")
                    
                    elif ctype == "agent_error":
                        agent_results = {"success": False, "error": chunk.get("error", "Error")}
                        print(f"   ❌ [AGENT] 에러: {agent_results['error']}")
                        new_history[-1][1] += f"\n❌ 오류: {agent_results['error']}\n"
                        break

                    elif ctype == "agent_complete":
                        agent_results = {"success": True, "content": chunk.get("content", ""), "tool_calls": chunk.get("tool_calls", [])}
                        print(f"   🤖 [AGENT] 완료. (도구 호출: {len(agent_results['tool_calls'])}개)")
                        break
                    
                    # 나머지 (thinking, tool logs)는 채팅창에 출력하지 않음 (pass)
                    else:
                        pass
            except Exception as e:
                logger.error(f"Agent stream error: {e}")
                agent_results = {"success": False, "error": str(e)}

            # Agent 결과를 LLM 컨텍스트에 추가하기 위해 포맷팅
            if agent_results.get("success"):
                agent_context = f"\n\n=== Agent 실행 결과 ===\n"
                for tool in agent_results.get("tool_calls", []):
                    agent_context += f"[{tool.get('name')}] {tool.get('result')}\n"
                if agent_results.get("content"):
                    agent_context += f"Agent 응답: {agent_results.get('content')}\n"

        # 3. [RAG Mode] 규정 검색 (키워드 매칭 및 예외 처리)
        # 단순 날씨 질문에 매뉴얼을 검색하는 과잉 개입 방지
        rag_keywords = ["규정", "지침", "매뉴얼", "절차", "기준", "행동요령", "위기관리", "대응", "조치"]
        weather_keywords = ["날씨", "미세먼지", "기온", "온도", "습도", "강수", "예보"]
        
        has_rag_kw = any(k in user_message for k in rag_keywords)
        is_weather_query = any(k in user_message for k in weather_keywords)
        
        # 규정 키워드가 있거나, 날씨 질문이 아닐 때만 RAG 수행 (단, 사용자가 enable_rag를 켰을 때)
        need_rag = has_rag_kw or (not is_weather_query and "특보" in user_message) or (not is_weather_query and "지진" in user_message)

        # 사용자가 명시적으로 RAG를 원하면(규정 키워드 포함) 무조건 실행
        if has_rag_kw: need_rag = True

        if enable_rag and need_rag:
            print(f"🚦 [TRACE] 라우팅 결정: >> [RAG 트랙] (키워드 매칭: {need_rag})")
            logger.info("🔍 [PROCESS] RAG 검색 수행")
            s_res = self.rag_system.search(user_message, selected_pdf_path, top_k=TOP_K_RESULTS)
            if s_res:
                print(f"   📚 [RAG] 검색 결과: {len(s_res)}건")
                # 검색 결과를 컨텍스트 문자열로 변환
                rag_context = "=== [행정안전부/기상청 공식 위기관리 매뉴얼 (SOP)] ===\n" + \
                              "\n".join([f"- {r['text']}" for r in s_res])
                
                # [FIX] 근거 자료 텍스트 생성
                rag_evidence_text = "\n\n---\n**[참고 자료 (규정/매뉴얼)]**\n"
                for r in s_res:
                    pdf_name = r.get('pdf_name', 'Unknown')
                    page_num = r.get('metadata', {}).get('page_num', '?')
                    score = r.get('similarity_score', 0.0)
                    rag_evidence_text += f"- 📄 **{pdf_name}** ({page_num}쪽) (관련도: {score:.2f})\n"
            else:
                print("   📚 [RAG] 검색 결과 없음")

        # 4. [Final Generation] 최종 응답 생성 및 스트리밍
        print("📨 [TRACE] 최종 프롬프트 조립 및 LLM 호출")
        if uploaded_context: print(f"   📎 [CTX] 업로드 파일 포함 ({len(uploaded_context)}자)")
        if rag_context: print(f"   📚 [CTX] RAG 컨텍스트 포함 ({len(rag_context)}자)")
        if agent_context: print(f"   🤖 [CTX] Agent 결과 포함 ({len(agent_context)}자)")
        
        messages = self._prepare_messages(user_message, history, rag_context, agent_context, uploaded_context)
        
        response = self.exaone_client.generate_response(
            messages=messages,
            stream=True,
            enable_thinking=reasoning_mode,
            temperature=self.reasoning_temperature if reasoning_mode else self.non_reasoning_temperature
        )
        
        # [Error Handling] API 호출 실패 시 방어 코드
        if response is None:
            print("❌ [TRACE] LLM API 호출 실패 (None)")
            logger.error("❌ EXAONE API 응답이 None입니다. (서버 연결 실패 또는 타임아웃)")
            new_history[-1][1] = "❌ 서버 연결에 실패했습니다. (API 응답 없음)\n관리자에게 문의하거나 잠시 후 다시 시도해주세요."
            yield (new_history, "")
            return

        # 스트리밍 청크 처리 (Thinking 파트와 Content 파트 분리)
        if reasoning_mode:
            for chunk in self._stream_response_with_reasoning(response):
                # UI 업데이트 (Thinking -> Content 순서)
                if chunk['type'] == 'thinking_chunk':
                    new_history[-1][1] += chunk['content']
                elif chunk['type'] == 'content':
                    new_history[-1][1] += chunk['content']
                yield (new_history, "")
        else:
            # 일반 스트리밍
            for chunk in self._stream_response_simple(response):
                new_history[-1][1] += chunk['content']
                yield (new_history, "")
        
        # [FIX] RAG 근거 자료 첨부
        if rag_evidence_text:
            new_history[-1][1] += rag_evidence_text
            yield (new_history, "")
        
        print("🏁 [TRACE] 처리 완료\n")

    def _stream_response_with_reasoning(self, response) -> Generator[dict, None, None]:
        """
        스트리밍 응답을 처리하며 <think> 태그를 감지하여 추론 과정과 최종 답변을 구분해 yield 합니다.
        UI에 실시간으로 사고 과정을 보여주기 위함입니다.
        """
        thinking_started = False
        content_started = False
        reasoning_content_complete = ""
        content_buffer = ""
        reasoning_seen = False
        
        try:
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta if hasattr(choice, 'delta') else {}
                    
                    # VLLM API 스펙에 따른 reasoning_content 필드 확인
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_chunk = delta.reasoning_content
                        reasoning_content_complete += reasoning_chunk
                        reasoning_seen = True
                        if not thinking_started:
                            yield {'type': 'thinking_start'}
                            thinking_started = True
                        yield {'type': 'thinking_chunk', 'content': reasoning_chunk}
                    
                    # 일반 content 필드 확인
                    if hasattr(delta, 'content') and delta.content:
                        content_chunk = delta.content
                        content_buffer += content_chunk
                        
                        # 추론이 있었고 이제 막 컨텐츠가 시작되는 경우
                        if reasoning_seen and not content_started and thinking_started:
                            yield {'type': 'thinking_end'}
                            content_started = True
                            content_buffer = ""
                            yield {'type': 'content', 'content': content_chunk}
                            continue
                        elif reasoning_seen and content_started:
                            yield {'type': 'content', 'content': content_chunk}
                            continue
                        
                        if content_started:
                            yield {'type': 'content', 'content': content_chunk}
                            continue
                        
                        # <think> 태그를 직접 파싱하여 처리 (API 필드가 아닌 텍스트 내 포함된 경우)
                        if '</think>' in content_buffer:
                            parts = content_buffer.split('</think>', 1)
                            thinking_part = parts[0]
                            answer_part = parts[1] if len(parts) > 1 else ""
                            
                            if not thinking_started:
                                yield {'type': 'thinking_start'}
                                thinking_started = True
                                if thinking_part: yield {'type': 'thinking_chunk', 'content': thinking_part}
                            
                            yield {'type': 'thinking_end'}
                            content_started = True
                            if answer_part: yield {'type': 'content', 'content': answer_part}
                            content_buffer = ""
                        else:
                            # </think>가 나오기 전까지는 추론 과정으로 간주
                            if not thinking_started:
                                yield {'type': 'thinking_start'}
                                thinking_started = True
                            yield {'type': 'thinking_chunk', 'content': content_chunk}
            
            # 스트림 종료 후 잔여 버퍼 처리
            if thinking_started and not content_started:
                yield {'type': 'thinking_end'}
                if reasoning_content_complete.strip():
                    cleaned_content = reasoning_content_complete
                    if '</think>' in cleaned_content:
                        cleaned_content = cleaned_content.split('</think>', 1)[1]
                    if cleaned_content.strip():
                        yield {'type': 'content', 'content': cleaned_content.strip()}
                elif content_buffer.strip():
                    yield {'type': 'content', 'content': content_buffer}
                            
        except Exception as e:
            logger.error(f"스트리밍 응답 처리 실패: {e}")
            yield {'type': 'error', 'content': f"스트리밍 처리 중 오류 발생: {e}"}
    
    def _stream_response_simple(self, response) -> Generator[dict, None, None]:
        """
        추론 모드가 아닐 때의 단순 스트리밍 처리입니다.
        [개선] 불필요한 버퍼링을 최소화하여 응답 속도 향상
        """
        think_tag_found = False
        content_buffer = ""
        try:
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    
                    # 1. reasoning_content가 있으면 즉시 처리 (드문 경우)
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        # 일반 모드에서는 추론 내용을 보여줄지 말지 정책에 따라 다름.
                        # 여기서는 content 취급하여 출력
                        yield {'type': 'content', 'content': delta.reasoning_content}
                    
                    # 2. content 처리
                    elif hasattr(delta, 'content') and delta.content:
                        if think_tag_found: 
                            # 이미 <think> 태그 처리가 끝났다면 무조건 즉시 출력
                            yield {'type': 'content', 'content': delta.content}
                        else:
                            content_buffer += delta.content
                            
                            # <think> 태그 감지 로직
                            if '</think>' in content_buffer:
                                think_tag_found = True
                                parts = content_buffer.split('</think>', 1)
                                if len(parts) > 1 and parts[1]: 
                                    yield {'type': 'content', 'content': parts[1]}
                                content_buffer = ""
                            elif '<think>' in content_buffer:
                                # 태그 시작됨, 닫힐 때까지 대기 (버퍼링 유지)
                                pass
                            elif '<' in content_buffer:
                                # 태그의 시작일 수도 있으니 잠시 대기 (단, 너무 길어지면 방출)
                                if len(content_buffer) > 50: 
                                    yield {'type': 'content', 'content': content_buffer}
                                    content_buffer = ""
                            else:
                                # 태그와 무관한 내용이면 즉시 방출! (속도 개선 핵심)
                                yield {'type': 'content', 'content': content_buffer}
                                content_buffer = ""
                                
            if content_buffer and not think_tag_found:
                yield {'type': 'content', 'content': content_buffer}
        except Exception as e:
            logger.error(f"Non-reasoning 스트리밍 처리 실패: {e}")
            yield {'type': 'error', 'content': f"스트리밍 처리 중 오류 발생: {e}"}

    def get_pdf_list(self) -> List[Tuple[str, str]]:
        """RAG 시스템에 로드된 PDF 목록을 반환합니다."""
        # rag_system에 해당 메서드가 구현되어 있다고 가정
        if hasattr(self.rag_system, 'get_pdf_list'):
            return self.rag_system.get_pdf_list()
        return []

