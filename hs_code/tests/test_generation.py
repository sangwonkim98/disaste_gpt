import logging
import sys
import os
from pathlib import Path

# PYTHONPATH 설정
sys.path.append(str(Path(__file__).parent.parent))

from core.generator import ReportGenerator

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def test_file_generation():
    print("🚀 [TEST] 보고서 파일 생성 테스트 시작")
    
    try:
        generator = ReportGenerator()
        file_path = generator.generate_daily_report_file()
        
        if "❌" in file_path:
            print(f"❌ 생성 실패: {file_path}")
        else:
            print(f"\n✅ 파일 생성 성공!")
            print(f"📂 파일 경로: {file_path}")
            
            # 파일 존재 여부 확인
            if os.path.exists(file_path):
                print(f"💾 파일 크기: {os.path.getsize(file_path)} bytes")
            else:
                print("❌ 파일이 실제로는 존재하지 않습니다.")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_file_generation()
