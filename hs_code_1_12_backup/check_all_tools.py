import sys
import json
import os
from pathlib import Path
import time

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

try:
    from services.agent_tools import exaone_agent_tools
    from config import KMA_API_KEY, get_conf
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

def run_test(tool_name, args, description):
    print(f"\n🧪 Testing: {tool_name}")
    print(f"   Desc: {description}")
    print(f"   Args: {args}")
    
    start_time = time.time()
    try:
        res_str = exaone_agent_tools.execute_tool(tool_name, args)
        duration = time.time() - start_time
        
        try:
            res = json.loads(res_str)
        except:
            print(f"   ❌ JSON 파싱 실패: {res_str[:100]}...")
            return False

        if "error" in res:
            print(f"   ❌ API 오류: {res['error']}")
            return False
        
        # 성공 케이스 분석
        data = res.get("data", [])
        if isinstance(data, list):
            count = len(data)
            print(f"   ✅ 성공 ({duration:.2f}s) - 데이터 {count}건")
            if count > 0:
                print(f"      Sample: {str(data[0])[:100]}...")
        else:
            print(f"   ✅ 성공 ({duration:.2f}s) - 데이터: {str(data)[:100]}...")
            
        return True, res # 결과 반환 (연쇄 테스트용)
        
    except Exception as e:
        print(f"   ❌ 실행 중 예외 발생: {e}")
        return False

def check_all():
    print("="*60)
    print("🛠️  EXAONE Agent Tools 전수 검사 시작")
    print("="*60)
    
    # 1. SerpAPI
    run_test("serpapi_web_search", {"query": "오늘 서울 날씨"}, "웹 검색")
    
    # 2. 초단기실황
    run_test("kma_get_ultra_srt_ncst", {"location": "서울"}, "기상청 초단기실황")
    
    # 3. 초단기예보
    run_test("kma_get_ultra_srt_fcst", {"location": "서울"}, "기상청 초단기예보")
    
    # 4. 단기예보 (Village) - 여기가 아까 문제였음
    run_test("kma_get_vilage_fcst", {"location": "서울"}, "기상청 단기예보")
    
    # 5. 중기 육상 예보
    # regId는 내부 매핑(서울=11B00000) 테스트
    run_test("kma_get_mid_land_fcst", {"location": "서울"}, "기상청 중기육상예보")
    
    # 6. 중기 기온 예보
    run_test("kma_get_mid_ta", {"location": "서울"}, "기상청 중기기온예보")
    
    # 7. 예비 특보
    run_test("kma_get_pwn_status", {}, "기상청 예비특보")
    
    # 8. 기상 특보
    run_test("kma_get_wthr_wrn_msg", {"stnId": "108"}, "기상청 기상특보")
    
    # 9. 지진 목록
    success, res = run_test("kma_get_eqk_msg_list", {}, "기상청 지진목록")
    
    # 10. 지진 상세 (목록이 있으면 첫번째 것으로 테스트)
    if success and res and res.get("data"):
        first_eqk = res["data"][0]
        tm_fc = first_eqk.get("tmFc")
        if tm_fc:
            print(f"      (지진 상세 조회를 위한 tmFc 추출: {tm_fc})")
            run_test("kma_get_eqk_msg", {"tmFc": tm_fc}, "기상청 지진상세")
    else:
        print("\nℹ️  지진 목록이 없으므로 상세 조회 테스트는 건너뜁니다.")

    print("\n" + "="*60)
    print("🏁  전수 검사 완료")
    print("="*60)

if __name__ == "__main__":
    check_all()
