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
        self.client = OpenAI(api_key=api_key, base_url=server_url)

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
        
        Returns: Generator yielding (history, doc_info)
        """
        
        # 1. [Special Flow] 보고서 생성 요청인지 확인
        if "보고서 생성" in user_message or "보고서 작성" in user_message:
            logger.info("📄 [PROCESS] 보고서 생성 요청 감지")
            # ReportGenerator 호출 로직 (별도 분기)
            # ... (생략: generator.py 참조) ...
            yield (history + [[user_message, "보고서 생성 로직 실행됨"]], "")
            return
        
        # 기본 변수 초기화
        rag_context = ""
        agent_context = None
        new_history = history + [[user_message, ""]]
        yield (new_history, "")

        # 2. [Agent Mode] 에이전트 실행 (툴 사용이 필요한 경우)
        if agent_mode and exaone_agent.is_available():
            # ... (Agent 실행 및 결과 수집 로직) ...
            pass

        # 3. [RAG Mode] 규정 검색 (키워드 매칭 시)
        rag_keywords = ["규정", "지침", "매뉴얼", "절차", "기준", "특보", "지진"]
        need_rag = any(k in user_message for k in rag_keywords)

        if enable_rag and need_rag:
            logger.info("🔍 [PROCESS] RAG 검색 수행")
            s_res = self.rag_system.search(user_message, selected_pdf_path, top_k=TOP_K_RESULTS)
            if s_res:
                # 검색 결과를 컨텍스트 문자열로 변환
                rag_context = "=== [행정안전부/기상청 공식 위기관리 매뉴얼 (SOP)] ===\n" + \
                              "\n".join([f"- {r['text']}" for r in s_res])

        # 4. [Final Generation] 최종 응답 생성 및 스트리밍
        messages = self._prepare_messages(user_message, history, rag_context, agent_context, uploaded_context)
        
        response = self.exaone_client.generate_response(
            messages=messages,
            stream=True,
            enable_thinking=reasoning_mode,
            temperature=self.reasoning_temperature if reasoning_mode else self.non_reasoning_temperature
        )
        
        # [Error Handling] API 호출 실패 시 방어 코드
        if response is None:
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
        """
        think_tag_found = False
        content_buffer = ""
        try:
            for chunk in response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    # reasoning_content가 있다면 일반 content로 취급하여 전달
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        if think_tag_found: yield {'type': 'content', 'content': delta.reasoning_content}
                        else:
                            content_buffer += delta.reasoning_content
                            if '</think>' in content_buffer:
                                think_tag_found = True
                                parts = content_buffer.split('</think>', 1)
                                if len(parts)>1 and parts[1]: yield {'type': 'content', 'content': parts[1]}
                                content_buffer = ""
                    elif hasattr(delta, 'content') and delta.content:
                        if think_tag_found: yield {'type': 'content', 'content': delta.content}
                        else:
                            content_buffer += delta.content
                            if '</think>' in content_buffer:
                                think_tag_found = True
                                parts = content_buffer.split('</think>', 1)
                                if len(parts)>1 and parts[1]: yield {'type': 'content', 'content': parts[1]}
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

