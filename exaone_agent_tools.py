"""
EXAONE 4.0 네이티브 Agentic Tool Use 구현
외부 API 호출을 위한 도구들
"""
import os
import json
import requests
import logging
from typing import Dict, List, Any, Optional
from config import KMA_API_KEY

logger = logging.getLogger(__name__)

class ExaoneAgentTools:
    """EXAONE 4.0 네이티브 tool use를 위한 도구 모음"""
    
    def __init__(self):
        self.tools = self._get_tools_definition()
        logger.info(f"EXAONE Agent 초기화 완료: {len(self.tools)}개 도구")
    
    def _get_tools_definition(self) -> List[Dict]:
        """EXAONE 4.0용 도구 정의 반환"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "serpapi_web_search",
                    "description": "SerpAPI를 사용해서 웹을 검색합니다. 최신 정보, 뉴스, 날씨, 사건 정보 등을 찾을 수 있습니다.",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "검색할 키워드나 질문"
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "kma_weather",
                    "description": "기상청 API를 사용해서 실시간 날씨 정보, 기상 경보, 예보 등을 조회합니다.",
                    "parameters": {
                        "type": "object",
                        "required": ["query"],
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "날씨 관련 질문 (예: '서울 날씨', '기상 경보')"
                            },
                            "location": {
                                "type": "string",
                                "description": "지역명 (기본값: 서울)",
                                "default": "서울"
                            }
                        }
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """도구 실행"""
        import time as _time
        logger.info(f"🔧 [TOOL] execute_tool 호출: tool_name={tool_name}, arguments={json.dumps(arguments, ensure_ascii=False)}")
        _start = _time.time()
        try:
            if tool_name == "serpapi_web_search":
                result = self._execute_serpapi_search(arguments)
            elif tool_name == "kma_weather":
                result = self._execute_kma_weather(arguments)
            else:
                result = f"알 수 없는 도구: {tool_name}"
            _elapsed = _time.time() - _start
            logger.info(f"🔧 [TOOL] execute_tool 완료: tool_name={tool_name}, 소요시간={_elapsed:.2f}초, 결과길이={len(result)}자")
            return result
        except Exception as e:
            _elapsed = _time.time() - _start
            logger.error(f"🚨 [TOOL] 도구 실행 실패 ({tool_name}): {e}, 소요시간={_elapsed:.2f}초")
            return f"도구 실행 중 오류 발생: {str(e)}"
    
    def _execute_serpapi_search(self, arguments: Dict[str, Any]) -> str:
        """SerpAPI 웹 검색 실행"""
        query = arguments.get("query", "")
        if not query:
            logger.warning("🔍 [SERPAPI] 검색어가 비어있음")
            return "검색어가 제공되지 않았습니다."

        api_key = os.getenv('SERPAPI_API_KEY')
        if not api_key:
            logger.warning("🔍 [SERPAPI] API 키 미설정")
            return "SerpAPI API 키가 설정되지 않았습니다."

        try:
            logger.info(f"🔍 [SERPAPI] 웹 검색 시작: query='{query}'")

            search_params = {
                "q": query,
                "api_key": api_key,
                "engine": "google",
                "hl": "ko",
                "gl": "kr",
                "num": 5,
                "safe": "active",
            }

            # API 키를 마스킹한 파라미터 로깅
            log_params = {k: v for k, v in search_params.items() if k != 'api_key'}
            log_params['api_key'] = '***MASKED***'
            logger.info(f"🔍 [SERPAPI] 요청 파라미터: {json.dumps(log_params, ensure_ascii=False)}")

            response = requests.get("https://serpapi.com/search.json", params=search_params, timeout=20)

            logger.info(f"🔍 [SERPAPI] HTTP 응답: status_code={response.status_code}")

            if response.status_code != 200:
                logger.error(f"🔍 [SERPAPI] API 요청 실패: status={response.status_code}, body={response.text[:300]}")
                return f"API 요청 실패 (상태 코드: {response.status_code})"

            data = response.json()
            results = data.get('organic_results', [])

            logger.info(f"🔍 [SERPAPI] 검색 결과: {len(results)}개 organic_results")

            if not results:
                logger.info(f"🔍 [SERPAPI] 검색 결과 없음. 응답 키: {list(data.keys())}")
                return "검색 결과를 찾을 수 없습니다."

            # 결과 포맷팅
            formatted_results = []
            for i, result in enumerate(results[:5], 1):
                title = result.get('title', '제목 없음')
                snippet = result.get('snippet', '설명 없음')
                link = result.get('link', '')

                logger.info(f"🔍 [SERPAPI] 결과 [{i}]: {title} | {link}")

                formatted_result = f"{i}. **{title}**\n   {snippet}\n   출처: {link}\n"
                formatted_results.append(formatted_result)

            result_text = "\n".join(formatted_results)
            logger.info(f"🔍 [SERPAPI] 검색 완료: {len(results)}개 결과, 포맷된 텍스트 {len(result_text)}자")

            return result_text

        except requests.exceptions.Timeout:
            logger.error(f"🔍 [SERPAPI] 타임아웃 발생 (20초)")
            return "검색 오류: 요청 시간 초과"
        except Exception as e:
            logger.error(f"🔍 [SERPAPI] 검색 실패: {type(e).__name__}: {e}")
            return f"검색 오류: {str(e)}"
    
    def _execute_kma_weather(self, arguments: Dict[str, Any]) -> str:
        """기상청 API 날씨 조회 실행"""
        query = arguments.get("query", "")
        location = arguments.get("location", "서울")

        api_key = KMA_API_KEY
        if not api_key or api_key == "YOUR_AUTH_KEY":
            logger.warning("🌤️ [KMA] 기상청 API 키 미설정")
            return "기상청 API 키가 설정되지 않았습니다."

        try:
            logger.info(f"🌤️ [KMA] 기상청 API 호출 시작: query='{query}', location='{location}'")

            # 현재 날씨 정보 조회
            logger.info("🌤️ [KMA] 1/3 현재 날씨 조회 중...")
            current_weather = self._get_current_weather(api_key, location)
            logger.info(f"🌤️ [KMA] 1/3 현재 날씨 결과: {current_weather[:200]}")

            # 기상 경보 정보
            logger.info("🌤️ [KMA] 2/3 기상 경보 조회 중...")
            weather_alerts = self._get_weather_alerts(api_key)
            logger.info(f"🌤️ [KMA] 2/3 기상 경보 결과: {weather_alerts[:200]}")

            # 예보 정보
            logger.info("🌤️ [KMA] 3/3 예보 정보 조회 중...")
            forecast = self._get_forecast(api_key, location)
            logger.info(f"🌤️ [KMA] 3/3 예보 결과: {forecast[:200]}")

            # 결과 조합
            result = f"""🌤️ **기상청 실시간 정보** (지역: {location})

**현재 날씨:**
{current_weather}

**기상 경보:**
{weather_alerts}

**예보 정보:**
{forecast}

*데이터 출처: 기상청 (data.kma.go.kr)*"""

            logger.info(f"🌤️ [KMA] 기상청 데이터 조회 완료: 전체 결과 {len(result)}자")
            return result

        except Exception as e:
            logger.error(f"🌤️ [KMA] 기상청 API 오류: {type(e).__name__}: {e}")
            return f"기상청 API 오류: {str(e)}"
    
    def _get_current_weather(self, api_key: str, location: str) -> str:
        """현재 날씨 정보 조회"""
        try:
            url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"

            from datetime import datetime
            now = datetime.now()
            base_date = now.strftime("%Y%m%d")
            base_time = now.strftime("%H00")

            params = {
                'serviceKey': api_key,
                'numOfRows': 10,
                'pageNo': 1,
                'dataType': 'JSON',
                'base_date': base_date,
                'base_time': base_time,
                'nx': '55',  # 서울 기준
                'ny': '127'
            }

            logger.info(f"  [KMA-현재날씨] URL: {url}")
            logger.info(f"  [KMA-현재날씨] params: base_date={base_date}, base_time={base_time}, nx=55, ny=127")

            response = requests.get(url, params=params, timeout=10)

            logger.info(f"  [KMA-현재날씨] HTTP status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                # 응답 헤더 로깅
                result_code = data.get('response', {}).get('header', {}).get('resultCode', 'N/A')
                result_msg = data.get('response', {}).get('header', {}).get('resultMsg', 'N/A')
                logger.info(f"  [KMA-현재날씨] API 응답코드: {result_code}, 메시지: {result_msg}")

                if 'response' in data and 'body' in data['response']:
                    items = data['response']['body'].get('items', {}).get('item', [])
                    logger.info(f"  [KMA-현재날씨] 수신 항목 수: {len(items)}")

                    weather_info = {}
                    for item in items:
                        category = item.get('category')
                        value = item.get('obsrValue')
                        logger.info(f"  [KMA-현재날씨] 항목: category={category}, obsrValue={value}")

                        if category == 'T1H':  # 기온
                            weather_info['기온'] = f"{value}°C"
                        elif category == 'REH':  # 습도
                            weather_info['습도'] = f"{value}%"
                        elif category == 'RN1':  # 강수량
                            weather_info['강수량'] = f"{value}mm"
                        elif category == 'WSD':  # 풍속
                            weather_info['풍속'] = f"{value}m/s"
                        elif category == 'PTY':  # 강수형태
                            weather_info['강수형태'] = self._get_precipitation_type(value)

                    if weather_info:
                        result = "\n".join([f"- {k}: {v}" for k, v in weather_info.items()])
                        logger.info(f"  [KMA-현재날씨] 파싱 결과: {weather_info}")
                        return result
                    else:
                        logger.warning("  [KMA-현재날씨] 날씨 항목 파싱 결과 비어있음")
                        return "현재 날씨 정보를 가져올 수 없습니다."
                else:
                    logger.warning(f"  [KMA-현재날씨] 응답 형식 오류: 키={list(data.get('response', {}).keys())}")
                    return "기상청 API 응답 형식이 올바르지 않습니다."
            else:
                logger.error(f"  [KMA-현재날씨] HTTP 실패: {response.status_code}, body={response.text[:300]}")
                return f"기상청 API 호출 실패 (HTTP {response.status_code})"

        except Exception as e:
            logger.error(f"  [KMA-현재날씨] 예외 발생: {type(e).__name__}: {e}")
            return f"현재 날씨 조회 오류: {str(e)}"
    
    def _get_weather_alerts(self, api_key: str) -> str:
        """기상 경보 정보 조회"""
        try:
            url = "http://apis.data.go.kr/1360000/WthrWrnInfoService/getWthrWrnMsg"

            params = {
                'serviceKey': api_key,
                'numOfRows': 10,
                'pageNo': 1,
                'dataType': 'JSON',
                'stnId': '108'  # 전국 기상 특보 (108: 전국)
            }

            logger.info(f"  [KMA-경보] URL: {url}")
            logger.info(f"  [KMA-경보] params: stnId=108 (전국)")

            response = requests.get(url, params=params, timeout=10)

            logger.info(f"  [KMA-경보] HTTP status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                # 응답 헤더 로깅
                result_code = data.get('response', {}).get('header', {}).get('resultCode', 'N/A')
                result_msg = data.get('response', {}).get('header', {}).get('resultMsg', 'N/A')
                logger.info(f"  [KMA-경보] API 응답코드: {result_code}, 메시지: {result_msg}")

                if 'response' in data and 'body' in data['response']:
                    items = data['response']['body'].get('items', {}).get('item', [])

                    # 단건 응답일 경우 리스트로 변환
                    if isinstance(items, dict):
                        logger.info(f"  [KMA-경보] 단건 응답 -> 리스트 변환")
                        items = [items]

                    logger.info(f"  [KMA-경보] 수신 항목 수: {len(items)}")

                    if items:
                        alerts = []
                        for idx, item in enumerate(items):
                            title = item.get('title', '')
                            msg = item.get('msg', '')
                            logger.info(f"  [KMA-경보] 항목 [{idx}]: title='{title}', msg 길이={len(msg)}자, keys={list(item.keys())}")
                            if title:
                                # msg가 너무 길 수 있으므로 요약
                                summary = msg[:200] + '...' if len(msg) > 200 else msg
                                alerts.append(f"- [{title}] {summary}")

                        if alerts:
                            logger.info(f"  [KMA-경보] 경보 {len(alerts)}건 발견")
                            return "\n".join(alerts)
                        else:
                            logger.info("  [KMA-경보] title이 있는 경보 항목 없음")
                            return "현재 발효 중인 기상 경보가 없습니다."
                    else:
                        logger.info("  [KMA-경보] items 비어있음 -> 경보 없음")
                        return "현재 발효 중인 기상 경보가 없습니다."
                else:
                    logger.warning(f"  [KMA-경보] 응답에 body 없음: 키={list(data.get('response', {}).keys())}")
                    # 원본 응답 일부 로깅 (디버깅용)
                    raw_text = json.dumps(data, ensure_ascii=False)[:500]
                    logger.warning(f"  [KMA-경보] 원본 응답(500자): {raw_text}")
                    return "기상 경보 정보를 가져올 수 없습니다."
            else:
                logger.error(f"  [KMA-경보] HTTP 실패: {response.status_code}, body={response.text[:300]}")
                return f"기상 경보 API 호출 실패 (HTTP {response.status_code})"

        except Exception as e:
            logger.error(f"  [KMA-경보] 예외 발생: {type(e).__name__}: {e}")
            return f"기상 경보 조회 오류: {str(e)}"
    
    def _get_forecast(self, api_key: str, location: str) -> str:
        """단기 예보 정보 조회"""
        try:
            url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"

            from datetime import datetime, timedelta
            now = datetime.now()
            base_date = now.strftime("%Y%m%d")
            base_time = "0500"

            params = {
                'serviceKey': api_key,
                'numOfRows': 100,
                'pageNo': 1,
                'dataType': 'JSON',
                'base_date': base_date,
                'base_time': base_time,
                'nx': '55',  # 서울 기준
                'ny': '127'
            }

            logger.info(f"  [KMA-예보] URL: {url}")
            logger.info(f"  [KMA-예보] params: base_date={base_date}, base_time={base_time}, nx=55, ny=127")

            response = requests.get(url, params=params, timeout=10)

            logger.info(f"  [KMA-예보] HTTP status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                result_code = data.get('response', {}).get('header', {}).get('resultCode', 'N/A')
                result_msg = data.get('response', {}).get('header', {}).get('resultMsg', 'N/A')
                logger.info(f"  [KMA-예보] API 응답코드: {result_code}, 메시지: {result_msg}")

                if 'response' in data and 'body' in data['response']:
                    items = data['response']['body'].get('items', {}).get('item', [])
                    logger.info(f"  [KMA-예보] 수신 항목 수: {len(items)}")

                    # 오늘과 내일 예보만 추출
                    today_forecast = []
                    tomorrow_forecast = []

                    for item in items:
                        fcst_date = item.get('fcstDate', '')
                        fcst_time = item.get('fcstTime', '')
                        category = item.get('category')
                        value = item.get('fcstValue')

                        if fcst_date == base_date:  # 오늘
                            if category == 'TMP':  # 기온
                                today_forecast.append(f"{fcst_time[:2]}시: {value}°C")
                        elif fcst_date == (datetime.strptime(base_date, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d"):  # 내일
                            if category == 'TMP':  # 기온
                                tomorrow_forecast.append(f"{fcst_time[:2]}시: {value}°C")

                    logger.info(f"  [KMA-예보] 오늘 예보: {len(today_forecast)}건, 내일 예보: {len(tomorrow_forecast)}건")

                    forecast_text = ""
                    if today_forecast:
                        forecast_text += f"오늘: {', '.join(today_forecast[:4])}\n"
                    if tomorrow_forecast:
                        forecast_text += f"내일: {', '.join(tomorrow_forecast[:4])}"

                    if forecast_text:
                        logger.info(f"  [KMA-예보] 예보 결과: {forecast_text}")
                        return forecast_text
                    else:
                        logger.warning("  [KMA-예보] TMP 카테고리 항목 없음")
                        return "예보 정보를 가져올 수 없습니다."
                else:
                    logger.warning(f"  [KMA-예보] 응답에 body 없음: 키={list(data.get('response', {}).keys())}")
                    return "예보 정보를 가져올 수 없습니다."
            else:
                logger.error(f"  [KMA-예보] HTTP 실패: {response.status_code}, body={response.text[:300]}")
                return f"예보 API 호출 실패 (HTTP {response.status_code})"

        except Exception as e:
            logger.error(f"  [KMA-예보] 예외 발생: {type(e).__name__}: {e}")
            return f"예보 조회 오류: {str(e)}"
    
    def _get_precipitation_type(self, pty_code: str) -> str:
        """강수형태 코드를 한글로 변환"""
        precipitation_types = {
            '0': '없음',
            '1': '비',
            '2': '비/눈',
            '3': '눈',
            '4': '소나기',
            '5': '빗방울',
            '6': '빗방울/눈날림',
            '7': '눈날림'
        }
        return precipitation_types.get(pty_code, '알 수 없음')

# 전역 인스턴스
exaone_agent_tools = ExaoneAgentTools()
