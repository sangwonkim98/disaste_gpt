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
from advanced_rag_system import AdvancedRAGSystem
from exaone_agent_manager import exaone_agent

logger = logging.getLogger(__name__)

class ExaoneClient:
    def __init__(self, server_url, api_key, model_name="LGAI-EXAONE/EXAONE-4.0-32B-GPTQ"):
        self.server_url = server_url
        self.api_key = api_key
        self.model_name = model_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=server_url,
        )

        logger.info(f"ExaoneClient initialized {server_url} with {model_name}")

    def generate_response(self, messages: List[Dict], enable_thinking: bool = False,
                          temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
                          top_p: float = TOP_P, stream: bool = True):

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

    def __init__(self):
        
        self.rag_system = AdvancedRAGSystem()

        self.exaone_client = ExaoneClient(
            server_url=VLLM_SERVER_URL,
            api_key=VLLM_API_KEY,
            model_name = LLM_MODEL_NAME,
        )

        self.reasoning_temperature = 0.6
        self.reasoning_top_p = 0.95
        self.non_reasoning_temperature = 0.4
        self.non_reasoning_top_p = 0.9

    def initialize_system(self, pdf_paths: List[str]):
        logger.info("시스템 초기화 시작...")
        self.rag_system.build_index(pdf_paths)
        logger.info("시스템 초기화 완료")
    
    def _prepare_messages(self, user_message: str, history: List[List[str]], rag_context: str = None, agent_context: str = None, reasoning_mode: bool = True):
        messages = []

        system_content = SYSTEM_MESSAGE
        # 비추론 모드에서는 <think> 태그 지시를 제거하여 불필요한 토큰 소비 방지
        if not reasoning_mode:
            system_content = re.sub(
                r'응답 형식:.*?한국어만 사용하세요\.',
                '응답 형식:\n구체적이고 실용적인 최종 답변을 한국어로 제공하세요.',
                system_content,
                flags=re.DOTALL
            )
        if agent_context:
            system_content += f"\n\n === Agent 실행 결과 ===\n{agent_context}\n\n위 결과를 참고하여 정확하고 유용한 답변을 제공하세요."
        if rag_context:
            system_content += f"\n\n === 참고 문서 ====\n{rag_context}\n\n위 문서 내용을 참고하여 정확하고 유용한 답변을 제공하세요."
        
        messages.append({
            "role": "system",
            "content": system_content
        })

        # 멀티턴 대화 컨텍스트 유지를 위한 히스토리 처리 개선
        # 히스토리가 너무 길면 최근 대화만 유지 (토큰 제한 고려)
        max_history_turns = 10  # 최대 10턴까지 유지
        if len(history) > max_history_turns:
            history = history[-max_history_turns:]
            logger.info(f"🔄 [HISTORY] 히스토리가 길어서 최근 {max_history_turns}턴만 유지")

        for human_msg, ai_msg in history:
            if human_msg and human_msg.strip():
                messages.append({"role": "user", "content": human_msg.strip()})
            if ai_msg and ai_msg.strip():
                # AI 메시지에서 UI 마크다운 제거하여 깔끔한 컨텍스트 유지
                clean_ai_msg = self._clean_ai_message(ai_msg.strip())
                messages.append({"role": "assistant", "content": clean_ai_msg})

        messages.append({"role": "user", "content": user_message})

        logger.info(f"🔧 [HISTORY] 준비된 메시지 수: {len(messages)} (시스템: 1, 히스토리: {len(history)*2}, 현재: 1)")
        return messages

    def _clean_ai_message(self, ai_msg: str) -> str:
        """AI 메시지에서 UI 마크다운과 상태 메시지를 제거하여 깔끔한 컨텍스트 유지"""
        # UI 상태 메시지 패턴 제거
        patterns_to_remove = [
            r'🔍 \[PROCESS\].*?\n?',
            r'🤖 EXAONE Agent.*?\n?',
            r'🤔 \*\*\[.*?\]\*\*\n?',
            r'💬 \*\*\[.*?\]\*\*\n?',
            r'```\n.*?\n```',  # 추론 과정 코드 블록
            r'✅.*?완료.*?\n?',
            r'❌.*?오류.*?\n?',
            r'🧠 \*\*\[.*?\]\*\*\n?',
            r'(?m)^\s*(🤖|🔧|⚙️|🧠|📱|✅|❌).*$\n?',
            r'🔧 .*?\n?',
            r'⚙️ .*?\n?',
            r'📱 .*?\n?',
            r'🧠 .*?\n?',
        ]
        
        cleaned_msg = ai_msg
        for pattern in patterns_to_remove:
            cleaned_msg = re.sub(pattern, '', cleaned_msg, flags=re.DOTALL | re.MULTILINE)
        
        # 연속된 공백과 줄바꿈 정리
        cleaned_msg = re.sub(r'\n\s*\n+', '\n\n', cleaned_msg)
        cleaned_msg = cleaned_msg.strip()
        
        return cleaned_msg

    def _estimate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 추정 (한국어 기준 대략 글자당 1.5토큰)"""
        if not text:
            return 0
        return int(len(text) * 1.5)

    def _estimate_messages_tokens(self, messages: List[Dict]) -> int:
        """메시지 리스트의 총 토큰 수를 추정"""
        total = 0
        for msg in messages:
            total += self._estimate_tokens(msg.get("content", ""))
            total += 4  # role 등 메타데이터 오버헤드
        return total

    def _truncate_rag_context(self, messages: List[Dict], max_input_tokens: int, max_tokens: int) -> List[Dict]:
        """입력 토큰이 한도를 초과하면 RAG 컨텍스트를 축소"""
        available_input = max_input_tokens - max_tokens - 100  # 여유분 100토큰
        estimated = self._estimate_messages_tokens(messages)

        if estimated <= available_input:
            return messages

        logger.warning(f"⚠️ [TRUNCATE] 입력 토큰 추정치({estimated}) > 허용 한도({available_input}), RAG 컨텍스트 축소")

        # system 메시지에서 RAG 컨텍스트 부분을 축소
        truncated = []
        for msg in messages:
            if msg["role"] == "system" and "=== 참고 문서 ====" in msg["content"]:
                parts = msg["content"].split("=== 참고 문서 ====")
                if len(parts) == 2:
                    base_content = parts[0]
                    rag_part = parts[1]

                    # 사용 가능한 토큰 수 계산
                    other_tokens = self._estimate_messages_tokens([m for m in messages if m["role"] != "system"])
                    base_tokens = self._estimate_tokens(base_content)
                    available_for_rag = available_input - other_tokens - base_tokens - 50

                    if available_for_rag > 200:
                        # RAG 컨텍스트를 허용 범위 내로 잘라냄
                        max_chars = int(available_for_rag / 1.5)
                        truncated_rag = rag_part[:max_chars]
                        logger.info(f"📝 [TRUNCATE] RAG 컨텍스트 축소: {len(rag_part)} -> {len(truncated_rag)} 문자")
                        msg = {"role": "system", "content": base_content + "=== 참고 문서 ====" + truncated_rag + "\n\n(일부 내용이 길이 제한으로 생략됨)"}
                    else:
                        # RAG 컨텍스트를 아예 제거
                        logger.warning("⚠️ [TRUNCATE] RAG 컨텍스트 완전 제거 (공간 부족)")
                        msg = {"role": "system", "content": base_content.rstrip() + "\n\n(참고 문서가 길이 제한으로 생략됨)"}
            truncated.append(msg)

        return truncated

    def _call_exaone_api(self, messages: List[Dict], stream, reasoning_mode):
        if reasoning_mode:
            temperature = self.reasoning_temperature
            top_p = self.reasoning_top_p
        else:
            temperature = self.non_reasoning_temperature
            top_p = self.non_reasoning_top_p

        enable_thinking = reasoning_mode

        # vLLM 서버의 max_model_len (config과 동기화 필요)
        max_model_len = 16384
        max_tokens = MAX_TOKENS if MAX_TOKENS <= max_model_len else 2048

        # 입력 토큰 초과 시 RAG 컨텍스트 자동 축소
        messages = self._truncate_rag_context(messages, max_model_len, max_tokens)

        # max_tokens 단계적 시도
        max_tokens_candidates = [max_tokens, max_tokens // 2, 1024]
        # 중복 제거 및 정렬
        max_tokens_candidates = sorted(set(max_tokens_candidates), reverse=True)

        for mt in max_tokens_candidates:
            logger.info(f"🔧 [TOKENS] max_tokens: {mt} 시도")
            response = self.exaone_client.generate_response(
                messages=messages,
                enable_thinking=enable_thinking,
                temperature=temperature,
                top_p=top_p,
                max_tokens=mt,
                stream=stream,
            )
            if response is not None:
                return response
            logger.warning(f"⚠️ [TOKENS] max_tokens={mt} 실패, 더 낮은 값으로 재시도")

        logger.error("❌ 모든 max_tokens 시도 실패")
        return None
    
    def _extract_exaone_response(self, response) -> str:
        if not response.strip():
            return "", ""
        
        # 케이스 1: <think>...</think> 형식
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, response, re.DOTALL)

        if think_match:
            thinking_part = think_match.group(1).strip()
            final_answer = re.sub(think_pattern, "", response, flags=re.DOTALL).strip()
            return thinking_part, final_answer
        
        # 케이스 2: </think>만 있는 형식 (추론 과정 -> </think> -> 최종 답변)
        think_end_pattern = r"(.*?)</think>(.*)"
        think_end_match = re.search(think_end_pattern, response, re.DOTALL)
        
        if think_end_match:
            thinking_part = think_end_match.group(1).strip()
            final_answer = think_end_match.group(2).strip()
            return thinking_part, final_answer
        
        # 케이스 3: 태그가 없는 경우
        return "", response.strip()
    
    def _print_final_response(self, reasoning_content, content):

        logger.info("\n" + "="*80)
        logger.info("🧠 EXAONE 4.0 REASONING & RESPONSE")
        logger.info("="*80)

        thinking_part = ""
        final_answer = ""

        extracted_thinking, extracted_final_answer = self._extract_exaone_response(reasoning_content)
        thinking_part = extracted_thinking if extracted_thinking else reasoning_content.strip()
        final_answer = content.strip() if content.strip() else extracted_final_answer

        logger.info("\n💡 추론 과정:")
        logger.info(thinking_part)
        logger.info("\n🔍 최종 답변:")
        logger.info(final_answer)
        logger.info("="*80 + "\n")

    def _stream_response_with_reasoning(self, response) -> Generator[dict, None, None]:
        """EXAONE reasoning을 지원하는 스트리밍 응답 처리"""
        thinking_started = False
        content_started = False
        reasoning_content_complete = ""  # 전체 reasoning_content 수집
        content_buffer = ""  # content에서 </think> 태그 감지용 버퍼
        reasoning_seen = False  # reasoning_content가 한 번이라도 도착했는지 추적
        
        try:
            # OpenAI 클라이언트 스트림 처리
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta if hasattr(choice, 'delta') else {}
                    
                    # EXAONE: reasoning_content 수집 (실시간 표시 + 최종 답변용)
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_chunk = delta.reasoning_content
                        reasoning_content_complete += reasoning_chunk  # 전체 수집
                        reasoning_seen = True
                        
                        if not thinking_started:
                            # 첫 번째 thinking chunk일 때 시작 신호
                            logger.info(f"🤔 [THINKING START] EXAONE Reasoning 시작")
                            yield {'type': 'thinking_start'}
                            thinking_started = True
                        yield {'type': 'thinking_chunk', 'content': reasoning_chunk}
                    
                    # content 필드 처리 (</think> 태그 감지)
                    if hasattr(delta, 'content') and delta.content:
                        content_chunk = delta.content
                        content_buffer += content_chunk
                        
                        # reasoning_content를 이미 받았다면 content는 곧바로 최종 답변으로 처리
                        if reasoning_seen and not content_started and thinking_started:
                            yield {'type': 'thinking_end'}
                            content_started = True
                            content_buffer = ""  # 답변 처리로 전환했으므로 버퍼 초기화
                            yield {'type': 'content', 'content': content_chunk}
                            continue
                        elif reasoning_seen and content_started:
                            # 이미 답변 구간이라면 바로 출력
                            yield {'type': 'content', 'content': content_chunk}
                            continue
                        
                        # 이미 content 모드면 바로 출력
                        if content_started:
                            yield {'type': 'content', 'content': content_chunk}
                            continue
                        
                        # </think> 태그 감지
                        if '</think>' in content_buffer:
                            logger.info(f"🔍 [CONTENT] content 필드에서 </think> 태그 감지")
                            
                            # </think> 전까지는 thinking으로, 이후는 content로 처리
                            parts = content_buffer.split('</think>', 1)
                            thinking_part = parts[0]
                            answer_part = parts[1] if len(parts) > 1 else ""
                            
                            # thinking이 시작 안됐으면 시작 신호 (버퍼에 쌓인 내용 출력)
                            if not thinking_started:
                                logger.info(f"🤔 [THINKING START] content에서 추론 과정 시작 (버퍼에서)")
                                yield {'type': 'thinking_start'}
                                thinking_started = True
                                # 버퍼에 쌓인 thinking 부분 한 번에 출력
                                if thinking_part:
                                    yield {'type': 'thinking_chunk', 'content': thinking_part}
                            # 이미 thinking이 시작됐으면 버퍼 내용을 다시 출력하지 않음 (중복 방지)
                            
                            yield {'type': 'thinking_end'}
                            content_started = True
                            
                            # </think> 이후 부분 출력
                            if answer_part:
                                yield {'type': 'content', 'content': answer_part}
                            
                            content_buffer = ""  # 버퍼 초기화
                        
                        else:
                            # </think> 태그가 아직 안나왔으면 thinking으로 실시간 출력
                            if not thinking_started:
                                logger.info(f"🤔 [THINKING START] content 필드에서 시작")
                                yield {'type': 'thinking_start'}
                                thinking_started = True
                            
                            yield {'type': 'thinking_chunk', 'content': content_chunk}
                    
                    elif hasattr(delta, 'content') and not delta.content:
                        # 빈 문자열이 오는 경우는 무시
                        pass
            
            # 스트림 종료 처리
            # finish_reason 확인 (마지막 chunk에서)
            if 'chunk' in dir() and hasattr(chunk, 'choices') and chunk.choices:
                finish_reason = getattr(chunk.choices[0], 'finish_reason', None)
                if finish_reason:
                    logger.info(f"🏁 [FINISH] finish_reason: {finish_reason}")
                    if finish_reason == 'length':
                        logger.warning("⚠️ [TRUNCATED] 토큰 한도로 인해 응답이 잘렸습니다!")
                        yield {'type': 'warning', 'content': '\n\n⚠️ (응답이 토큰 한도로 인해 잘렸습니다)'}

            if thinking_started and not content_started:
                logger.info(f"[DEBUG] Stream ended, sending thinking_end")
                yield {'type': 'thinking_end'}
                
                # content가 정말로 없는 경우에만 대체 content 출력
                # (content 필드에서 </think>로 구분된 경우는 이미 content_started=True이므로 여기 도달 안함)
                logger.warning(f"[DEBUG] content 없이 종료됨 - reasoning_content: {len(reasoning_content_complete)}자, buffer: {len(content_buffer)}자")
                
                # reasoning_content를 최종 답변으로 사용 (EXAONE non-reasoning 모드에서 발생)
                if reasoning_content_complete.strip():
                    # </think> 태그 제거 후 출력
                    cleaned_content = reasoning_content_complete
                    if '</think>' in cleaned_content:
                        parts = cleaned_content.split('</think>', 1)
                        cleaned_content = parts[1] if len(parts) > 1 else parts[0]
                    
                    if cleaned_content.strip():
                        yield {'type': 'content', 'content': cleaned_content.strip()}
                        content_started = True
                elif content_buffer.strip():
                    # 버퍼에만 내용이 있으면 버퍼 내용을 답변으로
                    yield {'type': 'content', 'content': content_buffer}
                    content_started = True
                            
        except Exception as e:
            logger.error(f"스트리밍 응답 처리 실패: {e}")
            yield {'type': 'error', 'content': f"스트리밍 처리 중 오류 발생: {e}"}

    def _stream_response_simple(self, response) -> Generator[dict, None, None]:
        """EXAONE Non-reasoning 모드용 스트리밍 응답 처리

        비추론 모드에서는 <think> 태그를 사용하지 않으므로,
        짧은 초기 버퍼로 </think> 존재 여부만 확인 후 즉시 스트리밍.
        """
        content_buffer = ""
        buffer_phase = True  # 초기 버퍼링 단계
        BUFFER_LIMIT = 50    # 첫 50자까지만 </think> 체크

        try:
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta if hasattr(choice, 'delta') else {}

                    # reasoning_content 또는 content에서 텍스트 추출
                    text = ""
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        text = delta.reasoning_content
                    elif hasattr(delta, 'content') and delta.content:
                        text = delta.content

                    if not text:
                        continue

                    if buffer_phase:
                        content_buffer += text
                        if '</think>' in content_buffer:
                            # </think> 발견 → 이후 내용만 출력
                            parts = content_buffer.split('</think>', 1)
                            buffer_phase = False
                            content_buffer = ""
                            if len(parts) > 1 and parts[1].strip():
                                yield {'type': 'content', 'content': parts[1]}
                        elif len(content_buffer) >= BUFFER_LIMIT:
                            # </think> 없음 → 버퍼 플러시 후 직접 스트리밍 전환
                            buffer_phase = False
                            yield {'type': 'content', 'content': content_buffer}
                            content_buffer = ""
                    else:
                        # 직접 스트리밍
                        yield {'type': 'content', 'content': text}

            # 스트림 종료 - 잔여 버퍼 플러시
            if content_buffer:
                yield {'type': 'content', 'content': content_buffer}

        except Exception as e:
            logger.error(f"Non-reasoning 스트리밍 처리 실패: {e}")
            yield {'type': 'error', 'content': f"스트리밍 처리 중 오류 발생: {e}"}

    # def _process_agent(self, user_message: str, history: List[List[str]], use_exaone_native: bool = True, reasoning_mode: bool = True) -> Dict:
    #     """Agent 처리를 간소화된 방식으로 실행"""
    #     # EXAONE 네이티브 tool use 우선 사용
    #     if use_exaone_native and exaone_agent.is_available():
    #         try:
    #             logger.info("🤖 EXAONE 네이티브 Agent 사용")
    #             agent_result = exaone_agent.run_agent(user_message, history, reasoning_mode)
                
    #             if agent_result.get("success"):
    #                 return {
    #                     "success": True,
    #                     "content": agent_result.get("content", ""),
    #                     "tool_calls": agent_result.get("tool_calls", []),
    #                     "agent_type": "exaone_native"
    #                 }
    #             else:
    #                 error_msg = agent_result.get("error", "EXAONE Agent 실행 실패")
    #                 logger.warning(f"⚠️ EXAONE Agent 실패: {error_msg}")
    #                 return {"success": False, "error": "Agent를 사용할 수 없습니다."}
                    
    #         except Exception as e:
    #             logger.error(f"❌ EXAONE Agent 예외: {e}")
    #             return {"success": False, "error": "Agent를 사용할 수 없습니다."}
    #     else:
    #         return {"success": False, "error": "EXAONE Agent를 사용할 수 없습니다."}

    def _reset_processing_state(self):
        """처리 상태 초기화 - 에러 복구나 새 세션 시작 시에만 사용"""
        logger.info("🔄 [RESET] ChatManager 상태 초기화 중...")
        
        # ExaoneClient 상태 초기화 (새 인스턴스 생성)
        try:
            self.exaone_client = ExaoneClient(
                server_url=VLLM_SERVER_URL,
                api_key=VLLM_API_KEY,
                model_name=LLM_MODEL_NAME,
            )
            logger.info("🔄 ExaoneClient 재초기화 완료")
        except Exception as e:
            logger.error(f"❌ ExaoneClient 재초기화 실패: {e}")
        
        # ExaoneAgentManager 상태도 초기화
        try:
            if hasattr(exaone_agent, 'client'):
                # 기존 클라이언트 상태 초기화
                from openai import OpenAI
                exaone_agent.client = OpenAI(
                    api_key=VLLM_API_KEY or "EMPTY",
                    base_url=VLLM_SERVER_URL
                )
                logger.info("🔄 ExaoneAgentManager 클라이언트 재초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ ExaoneAgentManager 초기화 실패: {e}")
    
    def start_new_session(self):
        """새로운 대화 세션 시작 - 명시적으로 상태를 초기화할 때 사용"""
        logger.info("🆕 [SESSION] 새로운 대화 세션 시작")
        self._reset_processing_state()
        logger.info("✅ [SESSION] 새로운 세션 준비 완료")

    def process_message(self, user_message: str, history: List[List[str]], agent_mode: bool = False, 
                    reasoning_mode: bool = True, enable_reasoning: bool = True, enable_rag: bool = True,
                    selected_pdf_path: Optional[str] = None, reset_state: bool = False) -> Generator[Tuple[List[List[str]], str], None, None]:
        
        # 필요한 경우에만 상태 초기화 (새 세션 시작, 에러 복구 등)
        if reset_state:
            logger.info("🔄 [RESET] 명시적 상태 초기화 요청")
            self._reset_processing_state()
        
        # 멀티턴 대화를 위한 컨텍스트 보존 - 매번 초기화하지 않음
        
        # 디버깅: 파라미터 상태 로깅
        
        logger.info(f"🔧 [DEBUG] ChatManager 파라미터 - Reasoning Mode: {reasoning_mode}, Enable Reasoning: {enable_reasoning}, Agent Mode: {agent_mode}, Enable RAG: {enable_rag}")
        
        if not user_message or not user_message.strip():
            yield (history, "❌ 빈 메시지는 처리할 수 없습니다.")
            return
        
        current_api_status = {"api": "", "message": ""}
        called_apis = []  

        def status_callback(api_name: str, message: str):
            current_api_status["api"] = api_name
            current_api_status["message"] = message
            
            # 호출된 API 추적 (더 정확한 패턴 매칭)
            if api_name == "serpapi":
                if any(keyword in message for keyword in ["웹 검색 완료", "검색 완료", "결과 발견"]):
                    if "SerpAPI" not in called_apis:
                        called_apis.append("SerpAPI")
                        logger.info(f"📱 [API-TRACK] SerpAPI 호출 확인: {message}")
            elif api_name == "kma":
                if any(keyword in message for keyword in ["기상청 데이터 조회 완료", "기상 정보", "날씨 정보"]):
                    if "기상청 API" not in called_apis:
                        called_apis.append("기상청 API")
                        logger.info(f"📱 [API-TRACK] 기상청 API 호출 확인: {message}")
            
            # 중요한 상태 변화만 출력 (reasoning과 progress는 출력 안함)
            if api_name not in ["reasoning", "progress"]:
                logger.info(f"🔄 [UI-STATUS] {api_name.upper()}: {message}")


        # selected_pdf_path는 매개변수로 받은 값 사용 (None이면 모든 PDF에서 검색)
        rag_context = ""
        search_results = []
        doc_info = ""

        try:
            logger.info(f"🔍 [PROCESS] 메시지 처리 시작: {user_message}")
            new_history = history + [[user_message, ""]]

            yield (new_history, "🔍 [PROCESS] 메시지 처리 시작")

            agent_results = {}
            agent_context = None

            if agent_mode and exaone_agent.is_available():
                logger.info("🤖 [PROCESS] Agent 모드 활성화")
                
                # Agent 모드에서는 항상 추론 모드 강제 활성화
                agent_reasoning_mode = True
                logger.info(f"🧠 [AGENT] Agent 모드에서 추론 모드 강제 활성화: {reasoning_mode} -> {agent_reasoning_mode}")
                
                # UI에 Agent 모드 시작 상태 표시
                new_history[-1][1] = "🤖 EXAONE Agent 실행 시작...\n\n"
                yield (new_history, doc_info)

                exaone_agent.set_status_callback(status_callback)

                agent_results = {"success": False}
                agent_status_message = ""
                agent_thinking_content = ""
                agent_thinking_in_progress = False
                
                try:
                    # Agent에 전달할 히스토리 클린업 (UI 마커, 상태 메시지 제거)
                    cleaned_history_for_agent = []
                    try:
                        for h_msg, a_msg in (history or []):
                            cleaned_agent_ai_msg = self._clean_ai_message(a_msg or "") if a_msg else ""
                            cleaned_history_for_agent.append([h_msg, cleaned_agent_ai_msg])
                    except Exception:
                        cleaned_history_for_agent = history or []

                    for result_chunk in exaone_agent.run_agent_stream(user_message, cleaned_history_for_agent, agent_reasoning_mode):
                        chunk_type = result_chunk.get("type")
                        chunk_message = result_chunk.get("message", "")

                        # Agent 추론 과정 스트리밍 처리 (Agent 모드에서는 항상 표시)
                        if chunk_type == "agent_thinking_start":
                            if enable_reasoning:  # Agent 모드에서는 사용자 추론 설정만 체크
                                agent_status_message = "🤖 EXAONE Agent 실행 시작...\n\n🤔 **[Agent 추론 과정]**\n```\n"
                                agent_thinking_in_progress = True
                                new_history[-1][1] = agent_status_message
                                yield (new_history, doc_info)
                        
                        elif chunk_type == "agent_thinking_chunk":
                            if enable_reasoning and agent_thinking_in_progress:
                                thinking_chunk = result_chunk.get("content", "")
                                agent_thinking_content += thinking_chunk
                                agent_status_message += thinking_chunk
                                new_history[-1][1] = agent_status_message
                                yield (new_history, doc_info)
                        
                        elif chunk_type == "agent_thinking_end":
                            if enable_reasoning and agent_thinking_in_progress:
                                agent_status_message += "\n```\n\n"
                                agent_thinking_in_progress = False
                                new_history[-1][1] = agent_status_message
                                yield (new_history, doc_info)
                        
                        # 최종 답변 추론 과정 처리 (Agent 모드에서는 항상 표시)
                        elif chunk_type == "agent_final_thinking_start":
                            if enable_reasoning:  # Agent 모드에서는 사용자 추론 설정만 체크
                                agent_status_message += "\n\n🧠 **[Agent 최종 답변 추론]**\n```\n"
                                agent_thinking_in_progress = True
                                new_history[-1][1] = agent_status_message
                                yield (new_history, doc_info)
                        
                        elif chunk_type == "agent_final_thinking_chunk":
                            if enable_reasoning and agent_thinking_in_progress:
                                thinking_chunk = result_chunk.get("content", "")
                                agent_status_message += thinking_chunk
                                new_history[-1][1] = agent_status_message
                                yield (new_history, doc_info)
                        
                        elif chunk_type == "agent_final_thinking_end":
                            if enable_reasoning and agent_thinking_in_progress:
                                agent_status_message += "\n```\n\n💬 **[Agent 최종 답변]**\n"
                                agent_thinking_in_progress = False
                                new_history[-1][1] = agent_status_message
                                yield (new_history, doc_info)
                        
                        # Agent content 스트리밍 처리
                        elif chunk_type == "content":
                            content_chunk = result_chunk.get("content", "")
                            agent_status_message += content_chunk
                            new_history[-1][1] = agent_status_message
                            yield (new_history, doc_info)
                        
                        # Agent 도구 호출 상태 처리
                        elif chunk_type == "tool_calls_start":
                            tool_count = result_chunk.get("tool_count", 0)
                            tool_msg = f"\n\n🔧 **[도구 호출 시작]** ({tool_count}개 도구)\n"
                            agent_status_message += tool_msg
                            new_history[-1][1] = agent_status_message
                            yield (new_history, doc_info)
                        
                        elif chunk_type == "tool_executing":
                            tool_name = result_chunk.get("tool_name", "unknown")
                            tool_msg = f"⚙️ {tool_name} 실행 중...\n"
                            agent_status_message += tool_msg
                            new_history[-1][1] = agent_status_message
                            yield (new_history, doc_info)
                        
                        elif chunk_type == "tool_complete":
                            tool_name = result_chunk.get("tool_name", "unknown")
                            tool_msg = f"✅ {tool_name} 완료\n"
                            agent_status_message += tool_msg
                            new_history[-1][1] = agent_status_message
                            yield (new_history, doc_info)

                        elif chunk_type == "final_response":
                            current_message = "\n\n🧠 Agent 결과를 바탕으로 최종 답변 생성 중...\n"
                            if not enable_reasoning:  # Agent 모드에서는 추론 항상 활성화되므로 enable_reasoning만 체크
                                agent_status_message = current_message
                            else:
                                agent_status_message += current_message
                            new_history[-1][1] = agent_status_message
                            yield (new_history, doc_info)
                            
                        elif chunk_type == "agent_complete":
                            agent_results = {
                                "success": True,
                                "content": result_chunk.get("content", ""),
                                "tool_calls": result_chunk.get("tool_calls", []),
                                "agent_type": "exaone_native"
                            }
                            # Agent 완료 메시지 표시
                            tool_count = len(agent_results.get("tool_calls", []))
                            completion_msg = f"\n\n✅ EXAONE Agent 완료! (도구 호출: {tool_count}개)\n\n"
                            if not enable_reasoning:  # Agent 모드에서는 추론 항상 활성화되므로 enable_reasoning만 체크
                                agent_status_message = completion_msg
                            else:
                                agent_status_message += completion_msg
                            break
                            
                        elif chunk_type == "agent_error":
                            agent_results = {"success": False, "error": result_chunk.get("error", "알 수 없는 오류")}
                            error_msg = f"❌ EXAONE Agent 오류: {agent_results['error']}\n\n"
                            if not enable_reasoning:  # Agent 모드에서는 추론 항상 활성화되므로 enable_reasoning만 체크
                                agent_status_message = error_msg
                            else:
                                agent_status_message += error_msg
                            break
                            
                except Exception as e:
                    logger.error(f"EXAONE Agent streaming 처리 실패: {e}")
                    agent_results = {"success": False, "error": str(e)}
                    agent_status_message = f"❌ EXAONE Agent 예외: {str(e)}\n\n"
                
            if agent_results.get("success"):
                logger.info("\n🔗 [UI] Agent 결과를 컨텍스트에 포함 중...")
                agent_context = f"\n\n=== Agent 실행 결과 ===\n"

                tool_calls = agent_results.get("tool_calls", [])
                if tool_calls:
                    logger.info(f"🔍 [UI] Agent 도구 호출 확인: {tool_calls}")
                    for tool in tool_calls:
                        tool_name = tool.get("name", "")
                        tool_result = tool.get("result", "")
                        agent_context += f"[{tool_name}] {tool_result}\n"
                        logger.info(f"🔍 [UI] Agent 도구 호출 결과: {tool_name} - {tool_result}")
                else:
                    logger.info("🔍 [UI] Agent 도구 호출 없음")

                
                agent_content = agent_results.get("content", "")
                if agent_content.strip():
                    agent_context += f"Agent 응답: {agent_content}\n"
                    logger.info(f"🔍 [UI] Agent 응답: {agent_content}")
                else:
                    logger.info("🔍 [UI] Agent 응답 없음")


            if enable_rag and user_message.strip():
                logger.info("🔍 [PROCESS] RAG 검색 모드 활성화")
                rag_status_message = "\n🔍 [PROCESS] RAG 검색 모드 활성화"
                existing_msg = new_history[-1][1] or ""
                new_history[-1][1] = f"{existing_msg}\n{rag_status_message}" if existing_msg else rag_status_message
                yield (new_history, doc_info)
                
                search_results = self.rag_system.search(
                    user_message,
                    selected_pdf_path,
                    top_k=TOP_K_RESULTS
                )

                if search_results:
                    rag_status_message = f"✅ RAG 검색 완료: {len(search_results)}개 결과"
                    existing_msg = new_history[-1][1] or ""
                    new_history[-1][1] = f"{existing_msg}\n{rag_status_message}" if existing_msg else rag_status_message
                    yield (new_history, doc_info)
                    # 검색 결과를 컨텍스트로 변환
                    contexts = []
                    for i, result in enumerate(search_results, 1):
                        context = f"[문서 {i}] {result['pdf_name']} (관련도: {result['similarity_score']:.3f})\n{result['text']}"
                        contexts.append(context)
                    
                    rag_context = "\\n\\n".join(contexts)
                    doc_info = self.rag_system.format_search_results(search_results, selected_pdf_path)
                    logger.info(f"✅ RAG 검색 완료: {len(search_results)}개 결과")

                else:
                    doc_info = "❌ 관련 문서를 찾을 수 없습니다."
                    print(f"❌ [UI] RAG 검색 결과 없음")
                    logger.warning("RAG 검색 결과 없음")
                    rag_status_message = "❌ RAG 검색 결과 없음"
                    existing_msg = new_history[-1][1] or ""
                    new_history[-1][1] = f"{existing_msg}\n{rag_status_message}" if existing_msg else rag_status_message
                    yield (new_history, doc_info)
            else:
                if not enable_rag:
                    print(f"\n📚 [UI] RAG 검색 비활성화됨")
                    doc_info = "📚 문서 내 검색이 비활성화되었습니다."
                else:
                    doc_info = "질문을 입력하면 관련 문서 내용이 여기에 표시됩니다."
            

            messages = self._prepare_messages(user_message, history, rag_context, agent_context, reasoning_mode=reasoning_mode)

                        # 최종 답변 생성 시작을 UI에 표시
            if agent_mode or enable_rag:
                final_status_message = "🧠 수집된 정보를 바탕으로 최종 답변 생성 중...\n\n"
            else:
                final_status_message = "\n\n🧠 질문에 대한 답변 생성 중...\n\n"
            
            new_history[-1][1] = final_status_message
            yield (new_history, doc_info)
            
            response = self._call_exaone_api(messages, stream=True, reasoning_mode=reasoning_mode)

            logger.info("\n💡 response~~~:")
            logger.info(response)



            accumulated_response = ""
            thinking_in_progress = False
            
            # 터미널 출력을 위한 별도 수집 (항상 수집)
            collected_reasoning = ""
            collected_content = ""
            
            # UI 표시용 변수 (reasoning 표시 여부에 따라)
            ui_accumulated_response = ""
            ui_content_started = False       

            print(f"\n🔧 [DEBUG-STREAM] 스트리밍 모드 결정:")
            print(f"  reasoning_mode: {reasoning_mode}")
            print(f"  enable_reasoning: {enable_reasoning}")
            print(f"  선택된 스트리밍 방식: {'_stream_response_with_reasoning' if reasoning_mode else '_stream_response_simple'}")
            
            if reasoning_mode:

                for chunk in self._stream_response_with_reasoning(response):
                    chunk_type = chunk.get('type')

                    if chunk_type == "thinking_start":
                        thinking_in_progress = True
                        if enable_reasoning:
                            if agent_mode:
                                existing_message = new_history[-1][1]
                                accumulated_response = existing_message + "\n\n🤔 **[추론 과정]**\n```\n"
                            else:
                                accumulated_response = "\n\n🤔 **[추론 과정]**\n```\n"
                            new_history[-1][1] = accumulated_response
                            logger.info(f"🤔 [UI] 추론 시작: {accumulated_response}")
                            yield (new_history, doc_info)

                    elif chunk_type == "thinking_chunk":
                        thinking_chunk = chunk.get("content", "")
                        collected_reasoning += thinking_chunk

                        if enable_reasoning and thinking_in_progress:
                            accumulated_response += thinking_chunk
                            new_history[-1][1] = accumulated_response
                            yield (new_history, doc_info)
                        elif not enable_reasoning:
                            # enable_reasoning이 false여도 수집은 하되 UI에는 표시 안함
                            pass
                    
                    elif chunk_type == "thinking_end":
                        if enable_reasoning and thinking_in_progress:
                            accumulated_response += "\n```\n\n💬 **[최종 답변]**\n"
                            new_history[-1][1] = accumulated_response
                            yield (new_history, doc_info)
                        thinking_in_progress = False
                    
                    elif chunk_type == "content":
                        content_chunk = chunk.get("content", "")
                        collected_content += content_chunk

                        if not enable_reasoning and not ui_content_started:
                            if agent_mode:
                                existing_message = new_history[-1][1]
                                accumulated_response = existing_message + "\n\n💬 **[최종 답변]**\n"
                            else:
                                accumulated_response = ""
                            ui_content_started = True
                        
                        accumulated_response += content_chunk
                        new_history[-1][1] = accumulated_response
                        yield (new_history, doc_info)
            else:
                for chunk in self._stream_response_simple(response):
                    chunk_type = chunk.get('type')
                    if chunk_type == "content":
                        content_chunk = chunk.get("content", "")
                        collected_content += content_chunk

                        if not ui_content_started:
                            if agent_mode:
                                existing_message = new_history[-1][1]
                                accumulated_response = existing_message + content_chunk
                            else:
                                accumulated_response = content_chunk
                            ui_content_started = True
                        else:
                            accumulated_response += content_chunk
                        new_history[-1][1] = accumulated_response
                        yield (new_history, doc_info)

            self._print_final_response(collected_reasoning, collected_content)

            if agent_mode:
                agent_formatted = exaone_agent.format_agent_result(agent_results)
                if agent_formatted:
                    existing_msg = new_history[-1][1] or ""
                    new_history[-1][1] = f"{existing_msg}{agent_formatted}"
                yield (new_history, doc_info)

            if new_history and len(new_history) > 0:
                final_message = new_history[-1][1]
                if final_message and final_message.strip():
                    # 정상적인 최종 응답이 있는 경우
                    logger.info(f"✅ 메시지 처리 완료 - 최종 응답 길이: {len(final_message)} 문자")
                    # 마지막에 한 번 더 yield하여 UI 업데이트 보장
                    yield (new_history, doc_info)
                else:
                    # 응답이 비어있는 경우 오류 처리
                    error_msg = "❌ 응답 생성이 완료되지 않았습니다. 다시 시도해주세요."
                    new_history[-1][1] = error_msg
                    logger.warning("⚠️ 응답이 비어있어서 오류 메시지로 대체")
                    yield (new_history, doc_info)
            else:
                # new_history가 없는 경우
                error_history = history + [[user_message, "❌ 응답을 생성할 수 없습니다."]]
                logger.error("❌ new_history가 생성되지 않음")
                yield (error_history, doc_info)
            
                logger.info("✅ 메시지 처리 완료")
            
        except Exception as e:
            logger.error(f"❌ 메시지 처리 실패: {e}")
            
            # 에러 발생 시 다음 요청을 위해 상태 재설정
            try:
                logger.info("🔄 [ERROR-RECOVERY] 에러 복구를 위한 상태 재설정 중...")
                self._reset_processing_state()
                logger.info("✅ [ERROR-RECOVERY] 상태 재설정 완료")
            except Exception as reset_error:
                logger.warning(f"⚠️ [ERROR-RECOVERY] 상태 재설정 실패: {reset_error}")
            
            # 사용자에게 친화적인 에러 메시지 제공
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                error_msg = "❌ 서버 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
            elif "token" in str(e).lower() or "api" in str(e).lower():
                error_msg = "❌ API 호출에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
            else:
                error_msg = f"❌ 처리 중 오류가 발생했습니다. 다시 시도해주세요.\n\n오류 세부사항: {str(e)}"
            
            error_history = history + [[user_message, error_msg]]
            yield (error_history, "⚠️ 오류가 발생했습니다. 상태가 재설정되었으니 다시 시도해주세요.")
    
    def get_pdf_list(self) -> List[Tuple[str, str]]:
        """PDF 목록 반환"""
        return self.rag_system.get_pdf_list()

if __name__ == "__main__":
    import logging
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO)








