from typing import List, Optional
from pydantic import BaseModel, Field

# ==========================================
# 데이터 스키마 (Schema) 정의
# ------------------------------------------
# 보고서의 각 부분이 어떤 형태(필드, 타입)를 가져야 하는지 정의합니다.
# Pydantic을 사용하면 LLM에게 "이 형식대로 JSON을 내놔"라고 강제하기 좋습니다.
# ==========================================

# --- 하위 컴포넌트 스키마 (부품들) ---

class WeatherInfo(BaseModel):
    """날씨 섹션 데이터 구조"""
    status_line: str = Field(..., description="전국 날씨 요약 (예: 전국 대체로 맑음, 기온 낮아 추움)")
    today_tomorrow_temp_line: str = Field(..., description="주요 도시 기온 정보 (예: (오늘) 서울 -5도 ...)")
    special_alerts: List[str] = Field(default=[], description="발효 중인 기상 특보 목록 (예: ['한파주의보', '강풍주의보'])")
    forecast_summary: str = Field(..., description="향후 기상 전망 요약")

class RiskSlot(BaseModel):
    """위험 요인 한 줄 요약"""
    slot: str = Field(..., description="위험 요인 카테고리 (한파, 강설, 빙판길, 건조, 강풍, 해상)")
    text: str = Field(..., description="해당 요인에 대한 요약 문구 (특이사항 없으면 '특이사항 없음' 또는 '-' 기재)")

class RiskSummary(BaseModel):
    """6대 위험 요인 종합 요약"""
    temperature_headline: Optional[str] = Field(None, description="기온/한파 관련 강조 헤드라인 (한파가 심각할 때만 생성)")
    slots: List[RiskSlot] = Field(..., description="6대 위험 요인 슬롯 리스트")

class Incident(BaseModel):
    """개별 사고/재난 정보"""
    category: str = Field(..., description="사고 유형 (화재, 교통, 붕괴 등)")
    sentence: str = Field(..., description="보고서용 개조식 문장 (예: (화재) 00:00경 서울 OO구 ...)")

class Operations(BaseModel):
    """대처 상황 및 통제 현황"""
    damage_status_line: str = Field("-", description="피해 현황 요약")
    control_status_line: str = Field("-", description="통제 상황 요약 (여객선, 도로 등)")
    action_items: List[str] = Field(default=[], description="주요 대처 상황 리스트 (부처명 포함)")

class ReportMeta(BaseModel):
    """보고서 메타데이터 (제목, 날짜 등)"""
    report_date_kor: str = Field(..., description="보고 날짜 (예: 2026. 1. 7.(수))")
    as_of_time: str = Field(..., description="기준 시간 (예: 06:00)")
    title: str = "국민 안전관리 일일상황"

# --- 최상위 보고서 스키마 ---

class DailyReport(BaseModel):
    """
    [최종 보고서 구조]
    이 모델이 전체 보고서의 '완성된 형태'입니다.
    Writer 노드가 최종적으로 이 구조를 채우는 것을 목표로 합니다.
    """
    meta: ReportMeta
    weather: WeatherInfo
    risk_summary: RiskSummary
    operations: Operations
    incidents: List[Incident] = Field(default=[])

    class Config:
        # JSON 변환 시 한글 깨짐 방지 힌트 등 설정 가능
        json_schema_extra = {
            "example": {
                "meta": {"report_date_kor": "2026. 1. 7.(수)", "as_of_time": "06:00", "title": "국민 안전관리 일일상황"},
                "weather": {"status_line": "...", "today_tomorrow_temp_line": "...", "special_alerts": [], "forecast_summary": "..."},
                "risk_summary": {"slots": [{"slot": "한파", "text": "..."}]},
                "operations": {"damage_status_line": "-", "control_status_line": "-", "action_items": []},
                "incidents": []
            }
        }
