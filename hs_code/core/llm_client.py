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
    def __init__(self, server_url, api_key, model_name="LGAI-EXAONE/EXAONE-4.0-32B"):
        self.server_url = server_url
        self.api_key = api_key
        self.model_name = model_name

        self.client = OpenAI(
            api_key=api_key,
            base_url=server_url,
        )

        logger.info(f"Report ExaoneClient initialized {server_url} with {model_name}")

    def generate_response(self, messages: List[Dict], enable_thinking: bool = False,
                          temperature: float = TEMPERATURE, max_tokens: int = MAX_TOKENS,
                          top_p: float = TOP_P, stream: bool = False):

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
