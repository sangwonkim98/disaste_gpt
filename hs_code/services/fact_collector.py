import logging
import json
from datetime import datetime
from typing import Dict, Any

# 기존 프로젝트의 도구 모듈 활용
from services.agent_tools import exaone_agent_tools

logger = logging.getLogger(__name__)

class FactCollector:
    """
    보고서 작성에 필요한 사실 데이터(Fact)를 강제로 수집하는 모듈
    LLM의 Tool Call을 기다리지 않고, 정해진 루틴대로 API를 호출함.
    """

    def __init__(self):
        self.tools = exaone_agent_tools

    def collect_all(self) -> Dict[str, str]:
        """날씨, 뉴스 등 모든 정보를 수집하여 딕셔너리로 반환"""
        
        logger.info("🔍 [FactCollector] 일일상황보고 작성을 위한 데이터 수집 시작")
        
        facts = {}
        
        # 1. 날씨 정보 수집 (강제 실행)
        try:
            weather_query = {"query": "전국 기상 특보 및 오늘 내일 날씨 전망", "location": "전국"}
            logger.info(f"☁️ 날씨 정보 수집 중... {weather_query}")
            weather_result = self.tools.execute_tool("kma_weather", weather_query)
            facts['weather_raw'] = weather_result
        except Exception as e:
            logger.error(f"❌ 날씨 수집 실패: {e}")
            facts['weather_raw'] = "날씨 정보를 가져올 수 없음."

        # 2. 뉴스/사건사고 수집 (강제 실행)
        try:
            # 어제~오늘 주요 재난/사고 키워드
            today_str = datetime.now().strftime("%Y년 %m월 %d일")
            search_query = {"query": f"{today_str} 주요 사건 사고 화재 교통사고 재난 뉴스"}
            
            logger.info(f"📰 뉴스 정보 수집 중... {search_query}")
            news_result = self.tools.execute_tool("serpapi_web_search", search_query)
            facts['news_raw'] = news_result
        except Exception as e:
            logger.error(f"❌ 뉴스 수집 실패: {e}")
            facts['news_raw'] = "뉴스 정보를 가져올 수 없음."

        # 3. (선택) 수집 시각 메타데이터
        facts['collected_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info("✅ [FactCollector] 데이터 수집 완료")
        return facts
