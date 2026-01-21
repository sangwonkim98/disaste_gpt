import logging
import json
from typing import List, Dict, Optional
from openai import OpenAI

from config import (
    VLLM_SERVER_URL, VLLM_API_KEY, LLM_MODEL_NAME, 
    MAX_TOKENS, TEMPERATURE, TOP_P
)

logger = logging.getLogger(__name__)

class ExaoneClient:
    """
    LLM 클라이언트 (EXAONE, Qwen3 등 vLLM 서빙 모델 호환)

    추론 모드 활성화:
    - EXAONE: extra_body의 chat_template_kwargs.enable_thinking
    - Qwen3: vLLM --enable-reasoning 플래그 + extra_body 동일 방식
    """

    def __init__(self, server_url, api_key, model_name="LGAI-EXAONE/EXAONE-4.0-32B"):
        self.server_url = server_url
        self.api_key = api_key
        self.model_name = model_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=server_url,
        )

        logger.info(f"LLMClient initialized: {server_url} with {model_name}")

    def generate_response(self, messages: List[Dict], enable_thinking: bool = False,
                          temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
                          top_p: float = TOP_P, stream: bool = False):
        """
        LLM 응답 생성

        Args:
            messages: 대화 메시지 리스트
            enable_thinking: 추론 모드 활성화 (True면 <think> 태그로 사고과정 출력)
            temperature: 샘플링 온도
            max_tokens: 최대 토큰 수
            top_p: nucleus sampling
            stream: 스트리밍 여부

        Returns:
            스트리밍이면 generator, 아니면 response 객체
        """
        try:
            # 추론 모드: extra_body로 enable_thinking 전달
            # EXAONE, Qwen3 (vLLM --enable-reasoning) 모두 이 방식 지원
            extra_body = {}
            if enable_thinking:
                extra_body["chat_template_kwargs"] = {"enable_thinking": True}

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                extra_body=extra_body if extra_body else None,
            )

            return response

        except Exception as e:
            logger.error(f"LLM API 호출 실패: {e}")
            return None
