import sys
import os
import json
from pathlib import Path

# 프로젝트 루트 경로 추가 (모듈 임포트를 위해)
sys.path.append(str(Path(__file__).parent))

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()

try:
    from services.agent_tools import exaone_agent_tools
    from config import KMA_API_KEY
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)

def test_kma_connectivity():
    print(f"\n🔑 KMA API KEY Check: {'✅ Found' if KMA_API_KEY else '❌ MISSING'}")
    if KMA_API_KEY:
        # 키 길이 등으로 간단한 유효성 체크 (보통 인코딩된 키는 길다)
        masked_key = KMA_API_KEY[:5] + "..." + KMA_API_KEY[-5:] if len(KMA_API_KEY) > 10 else "TOO_SHORT"
        print(f"   (Key: {masked_key})")
    else:
        print("   ⚠️  .env 파일에 KMA_API_KEY를 설정해주세요.")
        return

    print("\n📡 기상청 API 연결 테스트 시작...\n")

    # 1. 초단기 실황 (Current Weather)
    print("1️⃣ [초단기 실황] kma_get_ultra_srt_ncst (서울)")
    try:
        res_json = exaone_agent_tools.execute_tool("kma_get_ultra_srt_ncst", {"location": "서울"})
        data = json.loads(res_json)
        
        if "error" in data:
            print(f"   ❌ 실패: {data['error']}")
            if 'raw' in data:
                print(f"      [서버 응답] {data['raw'][:200]}...")
        else:
            base_time = data.get('base_time', 'N/A')
            items = data.get('data', [])
            print(f"   ✅ 성공! (기준시각: {base_time})")
            print(f"      데이터 개수: {len(items)}개")
            if items:
                # 첫 번째 아이템 예시 출력
                ex_item = items[0]
                print(f"      예시 데이터: {ex_item.get('category')} = {ex_item.get('obsrValue')}")
    except Exception as e:
        print(f"   ❌ 예외 발생: {e}")

    print("-" * 60)

    # 2. 단기 예보 (Village Forecast)
    print("2️⃣ [단기 예보] kma_get_vilage_fcst (서울)")
    try:
        res_json = exaone_agent_tools.execute_tool("kma_get_vilage_fcst", {"location": "서울"})
        data = json.loads(res_json)
        
        if "error" in data:
            print(f"   ❌ 실패: {data['error']}")
        else:
            base_time = data.get('base_time', 'N/A')
            items = data.get('data', [])
            item_count = data.get('item_count', 0)
            print(f"   ✅ 성공! (기준시각: {base_time})")
            print(f"      데이터 개수: {item_count}개")
    except Exception as e:
        print(f"   ❌ 예외 발생: {e}")

    print("-" * 60)

    # 3. 기상 특보 (Weather Warning)
    print("3️⃣ [기상 특보] kma_get_wthr_wrn_msg (전국/서울)")
    try:
        res_json = exaone_agent_tools.execute_tool("kma_get_wthr_wrn_msg", {"stnId": "108"})
        data = json.loads(res_json)
        
        if "error" in data:
            print(f"   ❌ 실패: {data['error']}")
        else:
            items = data.get('data', [])
            print(f"   ✅ 성공!")
            if not items:
                print("      ℹ️  현재 발효 중인 특보가 없습니다.")
            else:
                print(f"      ⚠️  발효 중인 특보 {len(items)}건")
                for item in items[:2]:
                    print(f"      - {item.get('title', '제목없음')}: {item.get('tmFc')}")
    except Exception as e:
        print(f"   ❌ 예외 발생: {e}")

    print("\n🏁 테스트 완료")

if __name__ == "__main__":
    test_kma_connectivity()
