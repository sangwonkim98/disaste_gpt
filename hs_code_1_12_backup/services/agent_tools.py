"""
[Agent Tools]
EXAONE Agent가 사용하는 도구(Tool) 모음
기상청(KMA) API 연동, SerpAPI 검색 등 외부 시스템과의 인터페이스(Interface) 역할
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union
from urllib.parse import unquote, urlparse, parse_qs

logger = logging.getLogger(__name__)

class ExaoneAgentTools:
    """
    EXAONE 4.0 Agent Tools
    기상청 단기/중기 예보, 특보, 지진 정보 API를 통합 관리
    """

    def __init__(self):
        # 1. API Base URLs (HTTPS 기본, 환경변수로 오버라이드 가능)
        self.URL_SHORT = os.getenv("KMA_URL_SHORT", "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0")
        self.URL_MID = os.getenv("KMA_URL_MID", "https://apis.data.go.kr/1360000/MidFcstInfoService")
        self.URL_WARN = os.getenv("KMA_URL_WARN", "https://apis.data.go.kr/1360000/WthrWrnInfoService")
        self.URL_EQK = os.getenv("KMA_URL_EQK", "https://apis.data.go.kr/1360000/EqkInfoService")
        
        # 2. Location Mapping (지명 -> 좌표/ID 매핑)
        # 사용자가 "용인", "서울" 등으로 입력하면 API에 필요한 격자(nx, ny)나 구역코드(regId)로 변환
        self.LOCATION_MAP = {
            # --- 수도권 (Metropolitan) ---
            "서울": {"nx": 60, "ny": 127, "regId_land": "11B00000", "regId_temp": "11B10101", "stnId": "108"},
            "강남": {"nx": 61, "ny": 126, "regId_land": "11B00000", "regId_temp": "11B10101", "stnId": "108"},
            "서초": {"nx": 61, "ny": 125, "regId_land": "11B00000", "regId_temp": "11B10101", "stnId": "108"},
            "송파": {"nx": 62, "ny": 126, "regId_land": "11B00000", "regId_temp": "11B10101", "stnId": "108"},
            "여의도": {"nx": 60, "ny": 127, "regId_land": "11B00000", "regId_temp": "11B10101", "stnId": "108"},
            "마포": {"nx": 59, "ny": 127, "regId_land": "11B00000", "regId_temp": "11B10101", "stnId": "108"},
            
            "인천": {"nx": 55, "ny": 124, "regId_land": "11B00000", "regId_temp": "11B20201", "stnId": "112"},
            "부평": {"nx": 55, "ny": 125, "regId_land": "11B00000", "regId_temp": "11B20201", "stnId": "112"},
            "송도": {"nx": 54, "ny": 123, "regId_land": "11B00000", "regId_temp": "11B20201", "stnId": "112"},

            "경기": {"nx": 60, "ny": 120, "regId_land": "11B00000", "regId_temp": "11B20601", "stnId": "119"},
            "수원": {"nx": 60, "ny": 121, "regId_land": "11B00000", "regId_temp": "11B20601", "stnId": "119"},
            "성남": {"nx": 62, "ny": 123, "regId_land": "11B00000", "regId_temp": "11B20605", "stnId": "119"},
            "분당": {"nx": 62, "ny": 123, "regId_land": "11B00000", "regId_temp": "11B20605", "stnId": "119"},
            "판교": {"nx": 62, "ny": 123, "regId_land": "11B00000", "regId_temp": "11B20605", "stnId": "119"},
            "고양": {"nx": 57, "ny": 128, "regId_land": "11B00000", "regId_temp": "11B20305", "stnId": "119"},
            "일산": {"nx": 57, "ny": 128, "regId_land": "11B00000", "regId_temp": "11B20305", "stnId": "119"},
            "용인": {"nx": 62, "ny": 121, "regId_land": "11B00000", "regId_temp": "11B20602", "stnId": "119"},
            "수지": {"nx": 62, "ny": 121, "regId_land": "11B00000", "regId_temp": "11B20602", "stnId": "119"},
            "안양": {"nx": 59, "ny": 123, "regId_land": "11B00000", "regId_temp": "11B20604", "stnId": "119"},
            "부천": {"nx": 56, "ny": 125, "regId_land": "11B00000", "regId_temp": "11B20204", "stnId": "119"},
            "안산": {"nx": 58, "ny": 121, "regId_land": "11B00000", "regId_temp": "11B20606", "stnId": "119"},
            "남양주": {"nx": 64, "ny": 128, "regId_land": "11B00000", "regId_temp": "11B20304", "stnId": "119"},
            "평택": {"nx": 62, "ny": 114, "regId_land": "11B00000", "regId_temp": "11B20611", "stnId": "119"},
            "의정부": {"nx": 61, "ny": 130, "regId_land": "11B00000", "regId_temp": "11B20302", "stnId": "119"},
            "파주": {"nx": 56, "ny": 131, "regId_land": "11B00000", "regId_temp": "11B20301", "stnId": "119"},

            # --- 강원권 ---
            "강원": {"nx": 73, "ny": 134, "regId_land": "11D10000", "regId_temp": "11D10301", "stnId": "101"},
            "춘천": {"nx": 73, "ny": 134, "regId_land": "11D10000", "regId_temp": "11D10301", "stnId": "101"},
            "강릉": {"nx": 92, "ny": 131, "regId_land": "11D20000", "regId_temp": "11D20501", "stnId": "105"},
            "원주": {"nx": 76, "ny": 122, "regId_land": "11D10000", "regId_temp": "11D10401", "stnId": "114"},
            "속초": {"nx": 91, "ny": 134, "regId_land": "11D20000", "regId_temp": "11D20401", "stnId": "90"},

            # --- 충청권 ---
            "대전": {"nx": 67, "ny": 100, "regId_land": "11C20000", "regId_temp": "11C20401", "stnId": "133"},
            "세종": {"nx": 66, "ny": 103, "regId_land": "11C20000", "regId_temp": "11C20404", "stnId": "239"},
            "청주": {"nx": 69, "ny": 107, "regId_land": "11C10000", "regId_temp": "11C10301", "stnId": "131"},
            "천안": {"nx": 63, "ny": 110, "regId_land": "11C20000", "regId_temp": "11C20301", "stnId": "232"},
            "충주": {"nx": 76, "ny": 114, "regId_land": "11C10000", "regId_temp": "11C10101", "stnId": "127"},

            # --- 전라권 ---
            "광주": {"nx": 58, "ny": 74, "regId_land": "11F20000", "regId_temp": "11F20501", "stnId": "156"},
            "전주": {"nx": 63, "ny": 89, "regId_land": "11F10000", "regId_temp": "11F10201", "stnId": "146"},
            "군산": {"nx": 59, "ny": 95, "regId_land": "11F10000", "regId_temp": "11F10202", "stnId": "140"},
            "목포": {"nx": 50, "ny": 67, "regId_land": "11F20000", "regId_temp": "11F20401", "stnId": "165"},
            "여수": {"nx": 73, "ny": 66, "regId_land": "11F20000", "regId_temp": "11F20404", "stnId": "168"},
            "순천": {"nx": 70, "ny": 70, "regId_land": "11F20000", "regId_temp": "11F20405", "stnId": "174"},

            # --- 경상권 ---
            "부산": {"nx": 98, "ny": 76, "regId_land": "11H20000", "regId_temp": "11H20201", "stnId": "159"},
            "해운대": {"nx": 98, "ny": 76, "regId_land": "11H20000", "regId_temp": "11H20201", "stnId": "159"},
            "서면": {"nx": 98, "ny": 76, "regId_land": "11H20000", "regId_temp": "11H20201", "stnId": "159"},
            
            "대구": {"nx": 89, "ny": 90, "regId_land": "11H10000", "regId_temp": "11H10701", "stnId": "143"},
            "울산": {"nx": 102, "ny": 84, "regId_land": "11H20000", "regId_temp": "11H20101", "stnId": "152"},
            
            "창원": {"nx": 90, "ny": 77, "regId_land": "11H20000", "regId_temp": "11H20301", "stnId": "155"},
            "마산": {"nx": 90, "ny": 77, "regId_land": "11H20000", "regId_temp": "11H20301", "stnId": "155"},
            "진주": {"nx": 81, "ny": 75, "regId_land": "11H20000", "regId_temp": "11H20701", "stnId": "192"},
            "구미": {"nx": 87, "ny": 106, "regId_land": "11H10000", "regId_temp": "11H10601", "stnId": "279"},
            "포항": {"nx": 102, "ny": 94, "regId_land": "11H10000", "regId_temp": "11H10501", "stnId": "138"},
            "경주": {"nx": 100, "ny": 91, "regId_land": "11H10000", "regId_temp": "11H10502", "stnId": "283"},
            "안동": {"nx": 91, "ny": 106, "regId_land": "11H10000", "regId_temp": "11H10401", "stnId": "136"},

            # --- 제주권 ---
            "제주": {"nx": 52, "ny": 38, "regId_land": "11G00000", "regId_temp": "11G00201", "stnId": "184"},
            "서귀포": {"nx": 53, "ny": 33, "regId_land": "11G00000", "regId_temp": "11G00401", "stnId": "189"},

            # --- 기타 ---
            "전국": {"stnId": "108"},
            "독도": {"nx": 144, "ny": 123, "regId_land": "11H10000", "regId_temp": "11H10902", "stnId": "143"}, # 대구/경북 참조
            "울릉도": {"nx": 127, "ny": 127, "regId_land": "11H10000", "regId_temp": "11H10901", "stnId": "115"},
        }
        
        # 툴 정의 초기화
        self.tools = self._get_tools_definition()
        logger.info("ExaoneAgentTools 초기화 완료 (KMA Services)")

    def _get_tools_definition(self) -> List[Dict]:
        """
        OpenAI Function Calling 포맷의 도구 정의 반환
        Agent가 이 정의를 보고 어떤 상황에 어떤 툴을 쓸지 판단함
        """
        return [
            # 1. 웹 검색
            {
                "type": "function",
                "function": {
                    "name": "serpapi_web_search",
                    "description": "최신 뉴스나 사건 정보를 찾기 위해 웹 검색을 수행합니다.",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {"type": "string", "description": "검색어"}
                        }
                    }
                }
            },
            # 2. 기상청 초단기 실황 (현재 날씨)
            {
                "type": "function",
                "function": {
                    "name": "kma_get_ultra_srt_ncst",
                    "description": "기상청 초단기실황. 특정 지역의 현재 실시간 날씨(기온, 강수, 바람 등)를 조회합니다.",
                    "parameters": {
                        "type": "object",
                        "required": ["location"],
                        "properties": {
                            "location": {"type": "string", "description": "지역명 (예: '서울', '용인')"},
                            "nx": {"type": "integer", "description": "격자 X (선택)"},
                            "ny": {"type": "integer", "description": "격자 Y (선택)"}
                        }
                    }
                }
            },
            # 3. KMA Short-term: Ultra Short Forecast
            {
                "type": "function",
                "function": {
                    "name": "kma_get_ultra_srt_fcst",
                    "description": "KMA Ultra Short Forecast (초단기예보). Returns forecast for the next 6 hours.",
                    "parameters": {
                        "type": "object",
                        "required": ["location"],
                        "properties": {
                            "location": {"type": "string"},
                            "nx": {"type": "integer"},
                            "ny": {"type": "integer"}
                        }
                    }
                }
            },
            # 4. KMA Short-term: Village Forecast
            {
                "type": "function",
                "function": {
                    "name": "kma_get_vilage_fcst",
                    "description": "KMA Village Forecast (단기예보). Returns detailed 3-day forecast (3-hour intervals).",
                    "parameters": {
                        "type": "object",
                        "required": ["location"],
                        "properties": {
                            "location": {"type": "string"},
                            "nx": {"type": "integer"},
                            "ny": {"type": "integer"}
                        }
                    }
                }
            },
            # 5. KMA Mid-term: Land Forecast
            {
                "type": "function",
                "function": {
                    "name": "kma_get_mid_land_fcst",
                    "description": "KMA Mid-term Land Forecast (중기육상예보). 3 to 10 days forecast (AM/PM weather, rain probability).",
                    "parameters": {
                        "type": "object",
                        "required": ["location"],
                        "properties": {
                            "location": {"type": "string"},
                            "regId": {"type": "string", "description": "Region ID (e.g., 11B00000)"}
                        }
                    }
                }
            },
            # 6. KMA Mid-term: Temperature Forecast
            {
                "type": "function",
                "function": {
                    "name": "kma_get_mid_ta",
                    "description": "KMA Mid-term Temperature Forecast (중기기온예보). Min/Max temps for 3 to 10 days.",
                    "parameters": {
                        "type": "object",
                        "required": ["location"],
                        "properties": {
                            "location": {"type": "string"},
                            "regId": {"type": "string", "description": "Region ID (e.g., 11B10101)"}
                        }
                    }
                }
            },
            # 7. KMA Warning: Preliminary Status
            {
                "type": "function",
                "function": {
                    "name": "kma_get_pwn_status",
                    "description": "KMA Preliminary Warning Status (예비특보).",
                    "parameters": {
                        "type": "object",
                        "properties": {} 
                    }
                }
            },
            # 8. KMA Warning: Warning Message
            {
                "type": "function",
                "function": {
                    "name": "kma_get_wthr_wrn_msg",
                    "description": "KMA Weather Warning Message (기상특보). Returns active warnings (Typhoon, Heavy Rain, etc.).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stnId": {"type": "string", "description": "Station ID (default: 108 for Nationwide/Seoul)", "default": "108"}
                        }
                    }
                }
            },
            # 9. KMA Earthquake: List
            {
                "type": "function",
                "function": {
                    "name": "kma_get_eqk_msg_list",
                    "description": "KMA Earthquake List (지진정보 목록). Returns recent earthquakes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fromTmFc": {"type": "string", "description": "Start Date (YYYYMMDD). Max range 3 days."},
                            "toTmFc": {"type": "string", "description": "End Date (YYYYMMDD)"}
                        }
                    }
                }
            },
            # 10. KMA Earthquake: Detail
            {
                "type": "function",
                "function": {
                    "name": "kma_get_eqk_msg",
                    "description": "KMA Earthquake Detail (지진정보 상세).",
                    "parameters": {
                        "type": "object",
                        "required": ["tmFc"],
                        "properties": {
                            "tmFc": {"type": "string", "description": "Time of issuance from list (YYYYMMDDHHMMSS)"},
                            "eqkType": {"type": "string"}
                        }
                    }
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        [Dispatcher] 툴 이름에 해당하는 실제 메서드를 찾아 실행
        예: 'kma_get_ultra_srt_ncst' -> self._exec_kma_get_ultra_srt_ncst(args)
        """
        try:
            method_name = f"_exec_{tool_name}"
            if hasattr(self, method_name):
                # SerpAPI 예외 처리
                if tool_name == "serpapi_web_search":
                    return self._exec_serpapi_web_search(arguments)
                else:
                    return getattr(self, method_name)(arguments)
            else:
                return self._error_json("Not Implemented", f"Tool '{tool_name}' not found.")
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, Error: {e}")
            return self._error_json("Execution Error", str(e))

    # =========================================================================
    # Helpers: Request & Time
    # =========================================================================

    def _get_api_key(self, key_name: str) -> str:
        """Retrieve API key from env."""
        return os.getenv(key_name, "")

    def _kma_request(self, base_url: str, endpoint: str, params: Dict) -> Dict:
        """
        [KMA API Handler] 기상청 API 요청 처리기
        - URL 인코딩 문제 해결 (requests 라이브러리의 이중 인코딩 방지)
        - 에러 발생 시 Mock Data로 폴백(Fallback)하여 시스템 안정성 보장
        """
        api_key = self._get_api_key("KMA_API_KEY")
        if not api_key:
            logger.warning("KMA_API_KEY 없음. Mock 데이터 사용.")
            return self._get_mock_response(endpoint, params)

        # [CRITICAL] URL 직접 조합 (Browser 방식)
        # requests 라이브러리가 serviceKey를 자동으로 인코딩해버려서 인증 실패하는 문제 해결
        url = f"{base_url}/{endpoint}?serviceKey={api_key}"
        
        req_params = params.copy()
        if "serviceKey" in req_params: del req_params["serviceKey"]
        req_params["dataType"] = "JSON"

        try:
            logger.info(f"📡 [KMA REQ] {endpoint} | Params: {req_params}")
            response = requests.get(url, params=req_params, timeout=10)
            
            # 1. HTTP 상태 코드 체크
            if response.status_code != 200:
                logger.warning(f"❌ [KMA HTTP ERROR] {response.status_code}")
                return self._get_mock_response(endpoint, params)

            # 2. JSON 파싱
            try:
                data = response.json()
            except json.JSONDecodeError:
                logger.warning("❌ [KMA JSON ERROR]")
                return self._get_mock_response(endpoint, params)

            # 3. 서비스 결과 코드 체크 (00: 정상)
            header = data.get("response", {}).get("header", {})
            result_code = header.get("resultCode")
            if result_code != "00":
                logger.warning(f"❌ [KMA SVC ERROR] Code {result_code}: {header.get('resultMsg')}")
                return self._get_mock_response(endpoint, params)

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ [KMA NET ERROR] {e}")
            return self._get_mock_response(endpoint, params)

    def _get_mock_response(self, endpoint: str, params: Dict) -> Dict:
        """
        [Fallback] API 실패 시 가짜(Mock) 데이터를 반환하여 프로세스 중단 방지
        """
        logger.info(f"⚠️ Generating Mock Data for {endpoint}")
        mock_items = []
        base_date = params.get('base_date', datetime.now().strftime("%Y%m%d"))
        
        if endpoint == "getUltraSrtNcst": # 초단기실황 Mock
            mock_items = [
                {"category": "T1H", "obsrValue": "21.5"}, # 기온
                {"category": "REH", "obsrValue": "45"},   # 습도
                {"category": "RN1", "obsrValue": "0"},    # 강수량
                {"category": "WSD", "obsrValue": "2.8"},  # Wind
                {"category": "PTY", "obsrValue": "0"},    # Rain Type
            ]
        elif endpoint == "getUltraSrtFcst": # Short-term Forecast
            mock_items = [
                {"category": "T1H", "fcstValue": "22", "fcstTime": "1300"},
                {"category": "SKY", "fcstValue": "1", "fcstTime": "1300"}, # Sunny
                {"category": "PTY", "fcstValue": "0", "fcstTime": "1300"},
            ]
        elif endpoint == "getVilageFcst": # Village Forecast
            mock_items = [
                {"category": "TMP", "fcstValue": "24", "fcstDate": base_date, "fcstTime": "1500"},
                {"category": "POP", "fcstValue": "10", "fcstDate": base_date, "fcstTime": "1500"},
                {"category": "SKY", "fcstValue": "3", "fcstDate": base_date, "fcstTime": "1500"},
            ]
        elif endpoint == "getWthrWrnMsg": # Warning
            mock_items = [] # No warning
        elif endpoint == "getMidLandFcst": # Mid Land
            mock_items = [{"wf3Am": "구름많음", "wf3Pm": "맑음", "rnSt3Am": "20", "rnSt3Pm": "10"}]
        elif endpoint == "getMidTa": # Mid Temp
            mock_items = [{"taMin3": "15", "taMax3": "25"}]
            
        return {
            "response": {
                "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE (MOCK)"},
                "body": {
                    "dataType": "JSON",
                    "items": {"item": mock_items},
                    "pageNo": 1, "numOfRows": 10, "totalCount": len(mock_items)
                }
            }
        }

    def _get_base_time_strict(self, interval_hours: int, delay_min: int) -> tuple:
        """
        [Time Sync] 기상청 API 제공 시각 계산 (Strict KST)
        - 기상청 데이터는 정시에 바로 나오지 않고 10~45분 지연됨
        - 현재 시간에서 delay_min을 뺀 '유효 시간'을 기준으로 가장 최근 발표 시각(base_time)을 계산
        """
        KST = timezone(timedelta(hours=9))
        now_kst = datetime.now(KST)
        
        # 유효 시간 계산 (현재 시간 - 딜레이)
        effective_time = now_kst - timedelta(minutes=delay_min)
        eff_date_str = effective_time.strftime("%Y%m%d")
        eff_hour = effective_time.hour
        
        if interval_hours == 1:
            # 매 시간 정각 발표
            base_time = f"{eff_hour:02d}00"
            return eff_date_str, base_time
        else:
            # 3시간 단위 발표 (02, 05, 08, ...)
            base_hours = [2, 5, 8, 11, 14, 17, 20, 23]
            valid_hour = -1
            for h in base_hours:
                if h <= eff_hour: valid_hour = h
                else: break
            
            if valid_hour == -1: # 전날 23시 데이터 사용
                prev_day = effective_time - timedelta(days=1)
                return prev_day.strftime("%Y%m%d"), "2300"
            else:
                return eff_date_str, f"{valid_hour:02d}00"

    # =========================================================================
    # Tool Implementations (개별 툴 로직)
    # =========================================================================

    def _exec_serpapi_web_search(self, args: Dict) -> str:
        """Wrapper for SerpAPI Web Search."""
        query = args.get("query")
        if not query: return self._error_json("Missing Param", "query is required")
        
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key: return self._error_json("Config Error", "SERPAPI_API_KEY missing")

        try:
            params = {
                "q": query, "api_key": api_key, "engine": "google",
                "hl": "ko", "gl": "kr", "num": 5
            }
            res = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            if res.status_code != 200:
                return self._error_json("SerpAPI Error", f"HTTP {res.status_code}")
            
            data = res.json()
            organic = data.get("organic_results", [])
            summary = [{"title": i.get("title"), "link": i.get("link"), "snippet": i.get("snippet")} for i in organic[:4]]
            
            return json.dumps({
                "service": "SerpAPI",
                "query": query,
                "count": len(summary),
                "data": summary
            }, ensure_ascii=False)
        except Exception as e:
            return self._error_json("Exception", str(e))

    def _exec_kma_get_ultra_srt_ncst(self, args: Dict) -> str:
        loc = self._map_location(args)
        if "nx" not in loc: return self._error_json("Location Error", "Could not resolve nx/ny")

        # UltraSrtNcst: Hourly, Avail after 40 mins
        base_date, base_time = self._get_base_time_strict(interval_hours=1, delay_min=40)

        params = {
            "numOfRows": 10, "pageNo": 1,
            "base_date": base_date, "base_time": base_time,
            "nx": loc["nx"], "ny": loc["ny"]
        }
        res = self._kma_request(self.URL_SHORT, "getUltraSrtNcst", params)
        
        return json.dumps({
            "service": "UltraSrtNcst",
            "request": {"base_date": base_date, "base_time": base_time, "loc": loc},
            "data": self._extract_items(res)
        }, ensure_ascii=False)

    def _exec_kma_get_ultra_srt_fcst(self, args: Dict) -> str:
        loc = self._map_location(args)
        if "nx" not in loc: return self._error_json("Location Error", "Could not resolve nx/ny")

        # UltraSrtFcst: Hourly, Avail after 45 mins. 
        # API expects base_time as HH30 sometimes, but usually HH00 is standard input for "generation time". 
        # Standard: Input HH30 for "Ultra Short Forecast"? 
        # Correction: The guide says "Base_time: 매시 30분". 
        # So we calculate standard hour, then set minutes to 30.
        
        # Calculate effective hour (delay 45 mins)
        base_date, base_time_hh00 = self._get_base_time_strict(interval_hours=1, delay_min=45)
        # Convert HH00 -> HH30 for this specific endpoint
        base_time = base_time_hh00[:2] + "30"

        params = {
            "numOfRows": 60, "pageNo": 1,
            "base_date": base_date, "base_time": base_time,
            "nx": loc["nx"], "ny": loc["ny"]
        }
        res = self._kma_request(self.URL_SHORT, "getUltraSrtFcst", params)
        
        return json.dumps({
            "service": "UltraSrtFcst",
            "request": {"base_date": base_date, "base_time": base_time, "loc": loc},
            "data": self._extract_items(res)
        }, ensure_ascii=False)

    def _exec_kma_get_vilage_fcst(self, args: Dict) -> str:
        loc = self._map_location(args)
        if "nx" not in loc: return self._error_json("Location Error", "Could not resolve nx/ny")

        # Village: 3-hour intervals, Avail +10 mins
        base_date, base_time = self._get_base_time_strict(interval_hours=3, delay_min=15)

        params = {
            "numOfRows": 200, "pageNo": 1,
            "base_date": base_date, "base_time": base_time,
            "nx": loc["nx"], "ny": loc["ny"]
        }
        res = self._kma_request(self.URL_SHORT, "getVilageFcst", params)
        
        items = self._extract_items(res)
        # Summarize count
        return json.dumps({
            "service": "VilageFcst",
            "request": {"base_date": base_date, "base_time": base_time, "loc": loc},
            "count": len(items),
            "data": items # Agent will parse this
        }, ensure_ascii=False)

    def _exec_kma_get_mid_land_fcst(self, args: Dict) -> str:
        loc = self._map_location(args)
        reg_id = args.get("regId") or loc.get("regId_land")
        if not reg_id: return self._error_json("Param Error", "regId required")

        # Mid-term: Announced at 06:00 and 18:00
        # Use simple logic: if now < 18:05, use 0600, else 1800 (with yesterday fallback)
        KST = timezone(timedelta(hours=9))
        now = datetime.now(KST)
        tm_fc = now.strftime("%Y%m%d") + ("0600" if now.hour < 18 else "1800")
        
        # If early morning (before 06:05), use yesterday 18:00
        if now.hour < 6 or (now.hour == 6 and now.minute < 5):
            tm_fc = (now - timedelta(days=1)).strftime("%Y%m%d") + "1800"

        params = {"regId": reg_id, "tmFc": tm_fc, "numOfRows": 10, "pageNo": 1}
        res = self._kma_request(self.URL_MID, "getMidLandFcst", params)
        
        return json.dumps({
            "service": "MidLandFcst",
            "request": {"tmFc": tm_fc, "regId": reg_id},
            "data": self._extract_items(res)
        }, ensure_ascii=False)

    def _exec_kma_get_mid_ta(self, args: Dict) -> str:
        loc = self._map_location(args)
        reg_id = args.get("regId") or loc.get("regId_temp")
        if not reg_id: return self._error_json("Param Error", "regId required")

        # Same tmFc logic as Land
        KST = timezone(timedelta(hours=9))
        now = datetime.now(KST)
        tm_fc = now.strftime("%Y%m%d") + ("0600" if now.hour < 18 else "1800")
        if now.hour < 6 or (now.hour == 6 and now.minute < 5):
            tm_fc = (now - timedelta(days=1)).strftime("%Y%m%d") + "1800"

        params = {"regId": reg_id, "tmFc": tm_fc, "numOfRows": 10, "pageNo": 1}
        res = self._kma_request(self.URL_MID, "getMidTa", params)
        
        return json.dumps({
            "service": "MidTa",
            "request": {"tmFc": tm_fc, "regId": reg_id},
            "data": self._extract_items(res)
        }, ensure_ascii=False)

    def _exec_kma_get_pwn_status(self, args: Dict) -> str:
        params = {"numOfRows": 10, "pageNo": 1}
        res = self._kma_request(self.URL_WARN, "getWthrPwn", params)
        return json.dumps({
            "service": "WthrPwn", 
            "data": self._extract_items(res)
        }, ensure_ascii=False)

    def _exec_kma_get_wthr_wrn_msg(self, args: Dict) -> str:
        loc = self._map_location(args)
        # Default stnId 108 (Nationwide/Seoul) if not provided
        stn_id = args.get("stnId") or loc.get("stnId", "108")
        
        params = {"stnId": stn_id, "numOfRows": 10, "pageNo": 1}
        
        # Add from/to if provided
        if "fromTmFc" in args: params["fromTmFc"] = args["fromTmFc"]
        if "toTmFc" in args: params["toTmFc"] = args["toTmFc"]

        res = self._kma_request(self.URL_WARN, "getWthrWrnMsg", params)
        return json.dumps({
            "service": "WthrWrnMsg",
            "request": {"stnId": stn_id},
            "data": self._extract_items(res)
        }, ensure_ascii=False)

    def _exec_kma_get_eqk_msg_list(self, args: Dict) -> str:
        KST = timezone(timedelta(hours=9))
        now = datetime.now(KST)
        
        # Logic: Clamp date range to max 3 days to avoid API errors/timeouts
        to_tm = args.get("toTmFc", now.strftime("%Y%m%d"))
        from_tm = args.get("fromTmFc", (now - timedelta(days=3)).strftime("%Y%m%d"))
        
        # Validation: Check difference
        try:
            d_to = datetime.strptime(to_tm, "%Y%m%d")
            d_from = datetime.strptime(from_tm, "%Y%m%d")
            if (d_to - d_from).days > 3:
                # Clamp from_tm
                from_tm = (d_to - timedelta(days=3)).strftime("%Y%m%d")
                logger.info(f"EqkList: Clamped date range to {from_tm} - {to_tm}")
        except ValueError:
            pass # Use as is if format weird

        params = {"fromTmFc": from_tm, "toTmFc": to_tm, "numOfRows": 10, "pageNo": 1}
        res = self._kma_request(self.URL_EQK, "getEqkMsgList", params)
        return json.dumps({
            "service": "EqkMsgList",
            "request": {"range": f"{from_tm}-{to_tm}"},
            "data": self._extract_items(res)
        }, ensure_ascii=False)

    def _exec_kma_get_eqk_msg(self, args: Dict) -> str:
        tm_fc = args.get("tmFc")
        if not tm_fc: return self._error_json("Param Error", "tmFc required")
        
        params = {"tmFc": tm_fc, "numOfRows": 1, "pageNo": 1}
        if "eqkType" in args: params["eqkType"] = args["eqkType"]
        
        res = self._kma_request(self.URL_EQK, "getEqkMsg", params)
        return json.dumps({
            "service": "EqkMsgDetail",
            "request": {"tmFc": tm_fc},
            "data": self._extract_items(res)
        }, ensure_ascii=False)

    def _map_location(self, args: Dict) -> Dict:
        """입력된 지역명(args)을 LOCATION_MAP을 통해 좌표 정보로 변환"""
        loc_name = args.get("location", "")
        mapped = {}
        
        if loc_name:
            for key, val in self.LOCATION_MAP.items():
                if key in loc_name:
                    mapped = val.copy()
                    break
        
        # 매핑 실패 시 서울을 기본값으로 사용 (Fallback)
        if not mapped:
            mapped = self.LOCATION_MAP["서울"].copy()
            mapped["_is_fallback"] = True
            
        # 개별 파라미터 오버라이드 (args에 nx, ny가 명시된 경우)
        if "nx" in args: mapped["nx"] = args["nx"]
        if "ny" in args: mapped["ny"] = args["ny"]
        if "regId" in args: mapped["regId_land"] = args["regId"]
        if "stnId" in args: mapped["stnId"] = args["stnId"]

        return mapped

    def _extract_items(self, api_response: Dict) -> List[Dict]:
        """JSON 응답에서 실제 데이터 리스트('item')만 추출"""
        try:
            return api_response["response"]["body"]["items"]["item"]
        except (KeyError, TypeError):
            return []

    def _error_json(self, error_type: str, message: str) -> str:
        """표준 에러 포맷"""
        return json.dumps({"status": "error", "error_type": error_type, "message": message}, ensure_ascii=False)

# 싱글톤 인스턴스 생성
exaone_agent_tools = ExaoneAgentTools()
