#!/usr/bin/env python3
"""
재난대응 대화형 인공지능 에이전트 실행 스크립트
EXAONE-4.0-32B-GPTQ + vLLM 서버 기반 RAG 챗봇
"""

import argparse
import logging
import sys
import colorlog
import requests
from pathlib import Path

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

# 환경 설정은 config.py에서 자동으로 실행됨
from config import GRADIO_HOST, GRADIO_PORT

try:
    from gradio_app import GradioApp, print_startup_info
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("💡 다음 명령어로 의존성을 설치하세요:")
    print("   pip install -r requirements.txt")
    sys.exit(1)

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """로깅 설정"""
    try:
        
        
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(getattr(logging, level.upper()))
        
    except ImportError:
        # colorlog가 없으면 기본 로깅 사용
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

def check_vllm_server():
    """vLLM 서버 연결 확인"""
    from config import VLLM_SERVER_URL
    
    try:
        response = requests.get(f"{VLLM_SERVER_URL}/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ vLLM 서버 연결 성공: {VLLM_SERVER_URL}")
            if models.get('data'):
                model_name = models['data'][0]['id']
                print(f"🤖 사용 가능한 모델: {model_name}")
                
                # EXAONE 모델인지 확인
                if "EXAONE" in model_name or "exaone" in model_name.lower():
                    print(f"✅ EXAONE 모델 확인됨: {model_name}")
                else:
                    print(f"⚠️  예상되지 않은 모델: {model_name}")
            return True
        else:
            print(f"❌ vLLM 서버 응답 오류: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ vLLM 서버 연결 실패: {e}")
        print("💡 외부 vLLM 서버가 실행 중인지 확인하세요.")
        return False

def check_system_requirements():
    """시스템 요구사항 확인"""
    print("🔍 시스템 요구사항 확인...")
    
    # 메모리 확인
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"💾 시스템 메모리: {memory_gb:.1f}GB")
        
        if memory_gb < 8:
            print("⚠️  최소 8GB 메모리가 권장됩니다.")
        else:
            print("✅ 메모리 요구사항 충족")
    except ImportError:
        print("⚠️  메모리 정보를 확인할 수 없습니다.")

def main():
    """메인 실행 함수"""
  
    parser = argparse.ArgumentParser(
        description='재난대응 대화형 인공지능 에이전트 (EXAONE-4.0-32B-GPTQ)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python run_server.py                    # 기본 실행
  python run_server.py --port 8080        # 포트 변경
  python run_server.py --share            # 공개 URL 생성
  python run_server.py --debug            # 디버그 모드
  python run_server.py --skip-checks      # 시스템 확인 생략
        """
    )
    
    parser.add_argument('--host', type=str, default=GRADIO_HOST, 
                       help=f'Host IP address (기본값: {GRADIO_HOST})')
    parser.add_argument('--port', type=int, default=GRADIO_PORT, 
                       help=f'Port number (기본값: {GRADIO_PORT})')
    parser.add_argument('--share', action='store_true', 
                       help='Create a public URL')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode')
    parser.add_argument('--skip-server-check', action='store_true', 
                       help='Skip vLLM server connectivity check')
    parser.add_argument('--skip-checks', action='store_true', 
                       help='Skip all system checks')
    
    args = parser.parse_args()
    
    # 로깅 설정
    log_level = "DEBUG" if args.debug else "INFO"
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    
    try:
        # 시작 정보 출력
        print_startup_info()
        print(f"📍 접속 URL: http://localhost:{args.port}")
        
        # 시스템 요구사항 확인
        if not args.skip_checks:
            check_system_requirements()
        
        # vLLM 서버 연결 확인
        if not args.skip_server_check and not args.skip_checks:
            if not check_vllm_server():
                print("\n❌ vLLM 서버 연결에 실패했습니다.")
                print("--skip-server-check 옵션으로 무시하고 실행하거나,")
                print("외부 vLLM 서버가 실행 중인지 확인하세요.")
                return 1
        
        print(f"{'='*80}")
        logger.info("🚀 재난대응 AI 에이전트 시작...")
        
        # Gradio 앱 생성 및 실행
        app = GradioApp()
        app.launch(host=args.host, port=args.port, share=args.share)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 사용자에 의해 중단됨")
        return 0
        
    except Exception as e:
        logger.error(f"❌ 애플리케이션 실행 실패: {e}")
        
        if args.debug:
            import traceback
            traceback.print_exc()
        
        print("\n💡 문제 해결 방법:")
        print("  1. 외부 vLLM 서버 확인:")
        print("     서버가 정상 작동 중인지 확인하세요.")
        print("  2. 의존성 설치:")
        print("     pip install -r requirements.txt")
        print("  3. PDF 파일 확인:")
        print("     data/ 디렉토리에 PDF 파일이 있는지 확인하세요.")
        print("  4. 디버그 모드:")
        print("     python run_server.py --debug")
        print("  5. 시스템 확인 생략:")
        print("     python run_server.py --skip-checks")
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 