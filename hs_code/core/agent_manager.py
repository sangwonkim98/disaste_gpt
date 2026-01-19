"""
EXAONE 4.0 네이티브 Agentic Tool Use 매니저
공식 tool calling 기능을 사용한 에이전트 구현
"""
import logging
import json
import re
from typing import List, Dict, Generator, Any, Optional
from openai import OpenAI
from config import VLLM_SERVER_URL, VLLM_API_KEY, LLM_MODEL_NAME, MAX_TOKENS
from services.agent_tools import exaone_agent_tools

logger = logging.getLogger(__name__)

class ExaoneAgentManager:
    """EXAONE 4.0 네이티브 tool use를 사용한 에이전트 매니저"""
    
    def __init__(self):
        """EXAONE Agent 초기화"""
        # OpenAI 클라이언트 초기화 (vLLM 서버와 호환)
        self.client = OpenAI(
            api_key=VLLM_API_KEY or "EMPTY",
            base_url=VLLM_SERVER_URL
        )
        
        # 도구 정의 가져오기
        self.tools = exaone_agent_tools.tools
        
        # 상태 콜백
        self.status_callback = None
        
        logger.info(f"EXAONE Agent 초기화 완료")
        logger.info(f"🤖 모델: {LLM_MODEL_NAME}")
        logger.info(f"🌐 서버: {VLLM_SERVER_URL}")
        logger.info(f"🔧 도구: {len(self.tools)}개")
        for tool in self.tools:
            logger.info(f"  - {tool['function']['name']}: {tool['function']['description'][:50]}...")
    
    def set_status_callback(self, callback):
        """상태 콜백 함수 설정"""
        self.status_callback = callback
    
    def _notify_status(self, api_name: str, message: str):
        """상태 알림"""
        if self.status_callback:
            self.status_callback(api_name, message)
    
    def is_available(self) -> bool:
        """Agent 사용 가능 여부 확인"""
        return True  # EXAONE 4.0은 항상 사용 가능
    
    def _extract_exaone_response(self, response: str) -> tuple:
        """
        EXAONE 응답에서 추론 과정과 최종 답변을 분리
        
        Returns:
            (thinking_part, final_answer) 튜플
        """
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
    
    def run_agent_stream(self, user_message: str, history: List[List[str]] = None, reasoning_mode: bool = True) -> Generator[Dict[str, Any], None, None]:
        """
        EXAONE 4.0 네이티브 tool use를 사용한 스트리밍 에이전트 실행
        
        이 함수는 다음과 같은 과정을 거칩니다:
        1. 시스템 프롬프트에 도구(Tool) 정의를 포함하여 LLM에게 전달
        2. LLM이 응답 생성 (Reasoning 모드 시 <think> 태그로 추론 과정 출력)
        3. LLM이 도구 사용을 요청하면(tool_calls), 해당 도구를 실행(execute_tool)
        4. 도구 실행 결과를 다시 LLM에게 전달하여 최종 답변 생성
        
        Args:
            user_message: 사용자 메시지
            history: 대화 히스토리
            reasoning_mode: 추론 모드 활성화 여부 (True: reasoning, False: non-reasoning)
        """
        try:
            self._notify_status("agent", "🤖 EXAONE Agent 실행 시작...")
            yield {
                "type": "status",
                "api": "agent",
                "message": "🤖 EXAONE Agent 실행 시작..."
            }
            
            # 메시지 구성
            messages = []
            
            # 시스템 메시지 (도구 사용 지침)
            # [FIX] 도구 목록 동적 생성 (하드코딩 제거)
            tool_list_str = "\n".join([f"- {t['function']['name']}: {t['function']['description']}" for t in self.tools])
            
            system_message = f"""당신은 도구를 사용할 수 있는 AI 어시스턴트입니다.
사용자의 질문에 답하기 위해 필요한 도구를 적절히 사용하세요.

사용 가능한 도구:
{tool_list_str}

지침:
1. 사용자의 질문을 해결하기 위해 가장 적절한 도구를 선택하세요.
2. 필요한 경우 여러 도구를 순차적으로 사용할 수 있습니다.
3. 도구 실행 결과가 나오면 이를 바탕으로 최종 답변을 작성하세요.
4. [중요] 날씨, 미세먼지, 뉴스 등 실시간 정보가 필요한 질문에는 반드시 도구를 사용해야 합니다. 도구를 사용하지 않고 임의로 날짜, 수치, 출처를 지어내지 마십시오.
5. 도구를 사용할 필요가 없는 일상적인 대화나 인사에는 자연스럽게 답변하세요.
"""
            
            messages.append({"role": "system", "content": system_message})
            
            # 대화 히스토리 (최근 2개만)
            if history:
                for human_msg, ai_msg in history[-2:]:
                    messages.append({"role": "user", "content": human_msg})
                    if ai_msg and ai_msg.strip():
                        messages.append({"role": "assistant", "content": ai_msg})
            
            # 현재 사용자 메시지
            messages.append({"role": "user", "content": user_message})
            
            self._notify_status("vllm", "🧠 EXAONE 서버로 요청 전송 중...")
            yield {
                "type": "status",
                "api": "vllm",
                "message": "🧠 EXAONE 서버로 요청 전송 중..."
            }
            
            # reasoning_mode에 따른 파라미터 설정
            # [OPTIMIZATION] Agent 도구 호출 단계에서는 속도를 위해 Thinking 비활성화
            # 사용자가 Reasoning을 켰더라도, 도구 선택은 빠르고 정확해야 하므로 False로 강제함
            if reasoning_mode:
                temperature = 0.4      # 도구 호출은 정확성이 중요하므로 온도를 낮춤
                top_p = 0.9           # 일반 모드와 동일하게 설정
                enable_thinking = False # [변경] True -> False (속도 향상)
                mode_name = "Agent-Fast (Reasoning requested but disabled for tool call)"
            else:
                temperature = 0.4      # EXAONE 공식 권장: non-reasoning mode
                top_p = 0.9           # 약간 낮춘 값
                enable_thinking = False
                mode_name = "Non-reasoning"
            
            logger.info(f"🎯 [DEBUG] EXAONE Agent 모드: {mode_name} (Temperature={temperature}, TopP={top_p}, Thinking={enable_thinking})")
            logger.info(f"🔧 [DEBUG] Reasoning Mode 입력 파라미터: {reasoning_mode} -> Enable Thinking: {enable_thinking}")
            
            # Agent 시작 알림
            yield {
                "type": "agent_start",
                "message": "🧠 EXAONE 서버로 요청 전송 중..."
            }
            
            # EXAONE 4.0 네이티브 tool use 호출
            response = self.client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=messages,
                tools=self.tools,  # EXAONE 4.0 네이티브 tool use
                tool_choice="auto",  # 자동 도구 선택
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                top_p=top_p,
                stream=True,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )
            
            # 스트리밍 응답 처리
            accumulated_content = ""
            accumulated_reasoning = ""
            tool_calls = []
            current_tool_call = None
            thinking_started = False
            content_started = False
            content_buffer = ""  # </think> 태그 감지용
            
            for chunk in response:
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Agent 추론 과정 스트리밍 처리 (reasoning_content)
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_chunk = delta.reasoning_content
                    accumulated_reasoning += reasoning_chunk
                    
                    if not thinking_started and reasoning_mode:
                        # 첫 번째 추론 청크일 때 시작 신호
                        yield {
                            "type": "agent_thinking_start",
                            "message": "🤔 EXAONE Agent 추론 시작..."
                        }
                        thinking_started = True
                    
                    if reasoning_mode:
                        yield {
                            "type": "agent_thinking_chunk",
                            "content": reasoning_chunk
                        }
                
                # 도구 호출 감지
                if delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        if tool_call_delta.index is not None:
                            # 새로운 도구 호출 시작 (리스트 확장)
                            if tool_call_delta.index >= len(tool_calls):
                                tool_calls.extend([None] * (tool_call_delta.index + 1 - len(tool_calls)))
                            
                            # 해당 인덱스의 도구 호출 객체가 없으면 초기화
                            if tool_calls[tool_call_delta.index] is None:
                                tool_calls[tool_call_delta.index] = {
                                    "id": "",  # 나중에 채워질 수 있음
                                    "type": "function",  # 기본값 설정
                                    "function": {
                                        "name": "",
                                        "arguments": ""
                                    }
                                }
                            
                            current_tool_call = tool_calls[tool_call_delta.index]
                            
                            # ID 업데이트 (보통 첫 청크에만 있음)
                            if tool_call_delta.id:
                                current_tool_call["id"] = tool_call_delta.id
                            
                            # Type 업데이트
                            if tool_call_delta.type:
                                current_tool_call["type"] = tool_call_delta.type
                            
                            # 도구 이름 추가
                            if tool_call_delta.function and tool_call_delta.function.name:
                                current_tool_call["function"]["name"] += tool_call_delta.function.name
                            
                            # 도구 인자 추가
                            if tool_call_delta.function and tool_call_delta.function.arguments:
                                current_tool_call["function"]["arguments"] += tool_call_delta.function.arguments
                
                # 일반 텍스트 내용 (</think> 태그 처리)
                if delta.content:
                    content_chunk = delta.content
                    accumulated_content += content_chunk
                    content_buffer += content_chunk
                    
                    # 이미 content 모드면 바로 출력
                    if content_started:
                        yield {
                            "type": "content",
                            "content": content_chunk
                        }
                        continue
                    
                    # </think> 태그 감지
                    if '</think>' in content_buffer:
                        logger.info(f"🔍 [AGENT] content 필드에서 </think> 태그 감지")
                        
                        parts = content_buffer.split('</think>', 1)
                        thinking_part = parts[0]
                        answer_part = parts[1] if len(parts) > 1 else ""
                        
                        # thinking이 시작 안됐으면 시작 신호 (버퍼에 쌓인 내용 출력)
                        if not thinking_started and reasoning_mode:
                            logger.info(f"🤔 [AGENT] 버퍼에서 추론 과정 시작")
                            yield {
                                "type": "agent_thinking_start",
                                "message": "🤔 EXAONE Agent 추론 시작..."
                            }
                            thinking_started = True
                            # 버퍼에 쌓인 thinking 부분 출력
                            if thinking_part:
                                yield {
                                    "type": "agent_thinking_chunk",
                                    "content": thinking_part
                                }
                        # 이미 thinking이 시작됐으면 (reasoning_content로 시작됨) 버퍼 내용을 다시 출력하지 않음
                        
                        # thinking_end 신호 (reasoning_mode일 때만)
                        if thinking_started and reasoning_mode:
                            yield {
                                "type": "agent_thinking_end",
                                "message": "🤔 EXAONE Agent 추론 완료"
                            }
                        thinking_started = False
                        content_started = True
                        
                        # </think> 이후 부분 출력
                        if answer_part:
                            yield {
                                "type": "content",
                                "content": answer_part
                            }
                        
                        content_buffer = ""
                    
                    else:
                        # </think> 태그가 아직 안나왔으면
                        if thinking_started and reasoning_mode:
                            # reasoning_content로 이미 thinking이 시작된 상태
                            # content가 오면 thinking 종료하고 content 모드로 전환
                            yield {
                                "type": "agent_thinking_end",
                                "message": "🤔 EXAONE Agent 추론 완료"
                            }
                            thinking_started = False
                            content_started = True
                            yield {
                                "type": "content",
                                "content": content_chunk
                            }
                        elif not thinking_started:
                            # thinking이 시작 안됐고 </think>도 없으면
                            if reasoning_mode:
                                # reasoning 모드면 thinking으로 처리
                                yield {
                                    "type": "agent_thinking_start",
                                    "message": "🤔 EXAONE Agent 추론 시작..."
                                }
                                thinking_started = True
                                yield {
                                    "type": "agent_thinking_chunk",
                                    "content": content_chunk
                                }
                            else:
                                # non-reasoning 모드면 바로 content로 출력
                                content_started = True
                                yield {
                                    "type": "content",
                                    "content": content_chunk
                                }
                        else:
                            # 기타 경우 content로 출력
                            yield {
                                "type": "content",
                                "content": content_chunk
                            }
            
            # 추론만 있고 content가 없는 경우 추론 종료 신호
            if thinking_started and reasoning_mode:
                yield {
                    "type": "agent_thinking_end", 
                    "message": "🤔 EXAONE Agent 추론 완료"
                }
                content_started = True
            
            # 도구 호출 실행
            tool_results = []
            if tool_calls:
                logger.info(f"🔧 도구 호출 처리 시작: {len(tool_calls)}개")
                
                # 도구 호출 시작 알림
                yield {
                    "type": "tool_calls_start",
                    "tool_count": len(tool_calls),
                    "message": f"🔧 도구 호출 처리 시작: {len(tool_calls)}개"
                }
                
                for i, tool_call in enumerate(tool_calls):
                    logger.info(f"🔧 도구 {i}: {tool_call}")
                    if tool_call and tool_call.get("function", {}).get("name"):
                        tool_name = tool_call["function"]["name"]
                        logger.info(f"🔧 도구 이름: {tool_name}")
                        try:
                            # JSON 인자 파싱
                            arguments = json.loads(tool_call["function"]["arguments"])
                            
                            # 도구 실행 시작 알림
                            self._notify_status(tool_name, f"🔧 {tool_name} 도구 실행 중...")
                            yield {
                                "type": "tool_executing",
                                "tool_name": tool_name,
                                "message": f"🔧 {tool_name} 도구 실행 중..."
                            }
                            
                            # 도구 실행
                            result = exaone_agent_tools.execute_tool(tool_name, arguments)
                            
                            # [안전장치] 결과가 문자열이 아니면 강제 형변환
                            if not isinstance(result, str):
                                try:
                                    result = json.dumps(result, ensure_ascii=False)
                                except:
                                    result = str(result)
                            
                            # [변경] 강제 절삭 제거 (Summarize Node에서 처리)
                            # if len(result) > 2000: ...
                            
                            # [디버깅] 도구 결과 로그 확인
                            logger.info(f"🔧 Tool Result ({tool_name}): {result[:200]}...")

                            tool_results.append({
                                "name": tool_name,
                                "result": result
                            })
                            
                            # 도구 실행 완료 알림
                            self._notify_status(tool_name, f"✅ {tool_name} 도구 실행 완료")
                            yield {
                                "type": "tool_complete",
                                "tool_name": tool_name,
                                "message": f"✅ {tool_name} 도구 실행 완료"
                            }
                        
                        except json.JSONDecodeError as e:
                            logger.error(f"도구 인자 JSON 파싱 실패: {e}")
                            tool_results.append({
                                "name": tool_name,
                                "result": f"도구 인자 파싱 오류: {str(e)}"
                            })
            
            # 도구 결과가 있으면 추가 응답 생성
            if tool_results:
                # 최종 응답 생성 시작 알림
                yield {
                    "type": "final_response", 
                    "message": "🧠 도구 결과를 바탕으로 최종 답변 생성 중..."
                }
                # 도구 결과를 메시지에 추가
                messages.append({
                    "role": "assistant",
                    "content": accumulated_content,
                    "tool_calls": tool_calls
                })
                
                # 도구 결과 추가
                for i, tool_result in enumerate(tool_results):
                    tool_call_id = None
                    if i < len(tool_calls) and tool_calls[i]:
                        tool_call_id = tool_calls[i].get("id")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result["result"]
                    })
                
                # 최종 응답 생성
                self._notify_status("vllm", "🧠 도구 결과를 바탕으로 최종 답변 생성 중...")
                yield {
                    "type": "status",
                    "api": "vllm",
                    "message": "🧠 도구 결과를 바탕으로 최종 답변 생성 중..."
                }
                
                # [DEBUG] vLLM 요청 전 Payload 검증
                # ID가 없는 tool_call이 있는지 확인 및 arguments 검증
                for msg in messages:
                    if msg.get("role") == "assistant" and "tool_calls" in msg:
                        for tc in msg["tool_calls"]:
                            # 1. ID 검증
                            if not tc.get("id"):
                                logger.warning(f"⚠️ [CRITICAL] Tool call ID missing! Tool: {tc}")
                                import uuid
                                tc["id"] = f"call_{str(uuid.uuid4())[:8]}"
                            
                            # 2. Arguments 검증 (빈 문자열이면 "{}"로)
                            if "function" in tc:
                                args = tc["function"].get("arguments")
                                if not args or not isinstance(args, str) or not args.strip():
                                    logger.warning(f"⚠️ [FIX] Empty arguments found for {tc['function'].get('name')}. Setting to '{{}}'")
                                    tc["function"]["arguments"] = "{}"

                # 메시지 로깅 (너무 길면 앞부분만)
                msg_log = json.dumps(messages, ensure_ascii=False, default=str)
                logger.info(f"📤 [vLLM REQ] Final Messages Payload (len={len(msg_log)}): {msg_log[:500]}...")

                final_response = self.client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=messages,
                    max_tokens=2048,
                    temperature=temperature,  # reasoning_mode에 따른 온도 사용
                    top_p=top_p,             # reasoning_mode에 따른 top_p 사용
                    stream=True,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": enable_thinking},
                    },
                )
                
                final_content = ""
                final_reasoning = ""
                final_thinking_started = False
                final_content_started = False
                final_content_buffer = ""
                
                for chunk in final_response:
                    if not chunk.choices:
                        continue
                    
                    delta = chunk.choices[0].delta
                    
                    # 최종 응답의 추론 과정 처리
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_chunk = delta.reasoning_content
                        final_reasoning += reasoning_chunk
                        
                        if not final_thinking_started and reasoning_mode:
                            yield {
                                "type": "agent_final_thinking_start",
                                "message": "🧠 최종 답변 추론 중..."
                            }
                            final_thinking_started = True
                        
                        if reasoning_mode:
                            yield {
                                "type": "agent_final_thinking_chunk",
                                "content": reasoning_chunk
                            }
                    
                    # 최종 응답 내용 (</think> 태그 처리)
                    if delta.content:
                        content_chunk = delta.content
                        final_content += content_chunk
                        final_content_buffer += content_chunk
                        
                        # 이미 content 모드면 바로 출력
                        if final_content_started:
                            yield {
                                "type": "content",
                                "content": content_chunk
                            }
                            continue
                        
                        # </think> 태그 감지
                        if '</think>' in final_content_buffer:
                            logger.info(f"🔍 [AGENT-FINAL] content 필드에서 </think> 태그 감지")
                            
                            parts = final_content_buffer.split('</think>', 1)
                            thinking_part = parts[0]
                            answer_part = parts[1] if len(parts) > 1 else ""
                            
                            # thinking이 시작 안됐으면 시작 신호 (버퍼에 쌓인 내용 출력)
                            if not final_thinking_started and reasoning_mode:
                                logger.info(f"🧠 [AGENT-FINAL] 버퍼에서 추론 과정 시작")
                                yield {
                                    "type": "agent_final_thinking_start",
                                    "message": "🧠 최종 답변 추론 중..."
                                }
                                final_thinking_started = True
                                # 버퍼에 쌓인 thinking 부분 출력
                                if thinking_part:
                                    yield {
                                        "type": "agent_final_thinking_chunk",
                                        "content": thinking_part
                                    }
                            # 이미 thinking이 시작됐으면 버퍼 내용을 다시 출력하지 않음 (중복 방지)
                            
                            if reasoning_mode:
                                yield {
                                    "type": "agent_final_thinking_end",
                                    "message": "🧠 최종 답변 추론 완료"
                                }
                            final_thinking_started = False
                            final_content_started = True
                            
                            # </think> 이후 부분 출력
                            if answer_part:
                                yield {
                                    "type": "content",
                                    "content": answer_part
                                }
                            
                            final_content_buffer = ""
                        
                        else:
                            # </think> 태그가 아직 안나왔으면
                            if final_thinking_started and reasoning_mode:
                                # reasoning_content로 이미 thinking이 시작된 상태
                                # content가 오면 thinking 종료하고 content 모드로 전환
                                yield {
                                    "type": "agent_final_thinking_end",
                                    "message": "🧠 최종 답변 추론 완료"
                                }
                                final_thinking_started = False
                                final_content_started = True
                                yield {
                                    "type": "content",
                                    "content": content_chunk
                                }
                            elif not final_thinking_started:
                                # thinking이 시작 안됐고 </think>도 없으면
                                if reasoning_mode:
                                    # reasoning 모드면 thinking으로 처리
                                    yield {
                                        "type": "agent_final_thinking_start",
                                        "message": "🧠 최종 답변 추론 중..."
                                    }
                                    final_thinking_started = True
                                    yield {
                                        "type": "agent_final_thinking_chunk",
                                        "content": content_chunk
                                    }
                                else:
                                    # non-reasoning 모드면 바로 content로 출력
                                    final_content_started = True
                                    yield {
                                        "type": "content",
                                        "content": content_chunk
                                    }
                            else:
                                # 기타 경우 content로 출력
                                yield {
                                    "type": "content",
                                    "content": content_chunk
                                }
                
                # 최종 추론만 있고 content가 없는 경우
                if final_thinking_started and reasoning_mode:
                    yield {
                        "type": "agent_final_thinking_end",
                        "message": "🧠 최종 답변 추론 완료"
                    }
                    final_content_started = True
                
                accumulated_content += final_content
            
            # [FIX] 도구 실행 결과 및 출처(Reference) 자동 첨부
            reference_text = ""
            if tool_results:
                reference_text = "\n\n---\n**[참고 자료 & 도구 실행 결과]**\n"
                for tool in tool_results:
                    t_name = tool.get("name", "")
                    t_res_str = tool.get("result", "")
                    
                    # 1. SerpAPI 검색 결과 파싱 (링크 추출)
                    if t_name == "serpapi_web_search":
                        try:
                            t_res_json = json.loads(t_res_str)
                            query = t_res_json.get("query", "")
                            reference_text += f"- 🔍 **웹 검색:** \"{query}\"\n"
                            
                            for item in t_res_json.get("data", []):
                                title = item.get("title", "No Title")
                                link = item.get("link", "#")
                                reference_text += f"  - [{title}]({link})\n"
                        except:
                            reference_text += f"- 🔍 **웹 검색:** (결과 파싱 실패)\n"

                    # 2. 기상청 API 등 기타 도구
                    elif "kma_" in t_name:
                        # JSON 파싱 시도
                        try:
                            t_res_json = json.loads(t_res_str)
                            svc_name = t_res_json.get("service", t_name)
                            reference_text += f"- 🔧 **기상청 API ({svc_name}):** 실행 완료\n"
                        except:
                            reference_text += f"- 🔧 **{t_name}:** 실행 완료\n"
                    
                    # 3. 그 외 도구
                    else:
                        reference_text += f"- ⚙️ **{t_name}:** 실행 완료\n"

            # 최종 결과 반환 (출처 포함)
            yield {
                "type": "agent_complete",
                "success": True,
                "content": accumulated_content + reference_text, # 원본 답변 + 출처
                "tool_calls": tool_results,
                "message": f"✅ EXAONE Agent 완료! (도구 호출: {len(tool_results)}개)"
            }
            
        except Exception as e:
            logger.error(f"EXAONE Agent 실행 실패: {e}")
            yield {
                "type": "agent_error",
                "success": False,
                "error": str(e),
                "message": f"❌ EXAONE Agent 오류: {str(e)}"
            }
    
    def run_agent(self, user_message: str, history: List[List[str]] = None, reasoning_mode: bool = True) -> Dict[str, Any]:
        """
        기존 방식의 Agent 실행 (하위 호환성)
        
        Args:
            user_message: 사용자 메시지
            history: 대화 히스토리
            reasoning_mode: 추론 모드 활성화 여부
        """
        final_result = None
        for chunk in self.run_agent_stream(user_message, history, reasoning_mode):
            if chunk.get("type") == "agent_complete":
                final_result = {
                    "success": chunk["success"],
                    "content": chunk["content"],
                    "tool_calls": chunk.get("tool_calls", [])
                }
            elif chunk.get("type") == "agent_error":
                final_result = {
                    "success": chunk["success"],
                    "error": chunk["error"]
                }
        
        return final_result or {
            "success": False,
            "error": "EXAONE Agent 실행 중 알 수 없는 오류가 발생했습니다."
        }
    
    def format_agent_result(self, agent_result: Dict[str, Any]) -> str:
        """Agent 결과 포맷팅"""
        if not agent_result.get("success"):
            return f"\n\n**🤖 EXAONE Agent 오류:** {agent_result.get('error', '알 수 없는 오류')}"
        
        # 도구 호출 결과 표시
        tool_calls = agent_result.get("tool_calls", [])
        tool_results = []
        
        for tool in tool_calls:
            tool_name = tool.get("name", "unknown")
            tool_result = tool.get("result", "")
            
            if tool_name == "serpapi_web_search" and tool_result:
                tool_results.append(f"\n\n**🔍 웹 검색 결과:**\n{tool_result}")
            elif tool_name == "kma_weather" and tool_result:
                tool_results.append(f"\n\n**🌤️ 기상 정보:**\n{tool_result}")
        
        # Agent 응답 내용
        #content = agent_result.get("content", "")

        # print("================================================")
        # print("content", content)
        # print("================================================")
        
        # 결과 조합
        formatted_result = "".join(tool_results)
        return formatted_result

# 전역 인스턴스
exaone_agent = ExaoneAgentManager()
