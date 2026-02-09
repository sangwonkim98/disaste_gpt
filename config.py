"""
재난대응 AI 에이전트 설정
.env 파일에서 환경변수를 로드
"""

import os
import glob
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ================================
# 프로젝트 정보
# ================================
PROJECT_NAME = "재난대응 대화형 인공지능 에이전트"
VERSION = "v5.0"
DESCRIPTION = "EXAONE 4.0 + vLLM + RAG 기반 재난대응 챗봇"

# ================================
# 경로 설정
# ================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"
VECTORSTORE_CACHE_DIR = CACHE_DIR / "vectorstore"
OCR_OUTPUT_DIR = CACHE_DIR / "ocr"
TEMP_DIR = PROJECT_ROOT / "temp"
MODEL_CACHE_DIR = str(CACHE_DIR / "models")

# ================================
# vLLM 서버 설정 (.env에서 로드)
# ================================
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

# ================================
# 모델 설정 (.env에서 로드)
# ================================
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "LGAI-EXAONE/EXAONE-4.0-32B-GPTQ")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
ENABLE_REASONING = True

# EXAONE 4.0 공식 권장 파라미터
TEMPERATURE = 0.6  # reasoning 모드
TOP_P = 0.95
TOP_K = 20
MIN_P = 0

# ================================
# 임베딩 모델
# ================================
EMBEDDING_MODEL = "dragonkue/BGE-m3-ko"

# ================================
# RAG 설정
# ================================
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 3000
TOP_K_RESULTS = 3

# Query Rewriting
QUERY_REWRITING_ENABLED = True
QUERY_REWRITE_NUM = 3
QUERY_REWRITE_TIMEOUT = 25

# Reranking
RERANKING_ENABLED = True
RERANK_TOP_N = 24
CANDIDATES_PER_QUERY = 12
ORIGINAL_QUERY_WEIGHT = 0.6
REWRITE_QUERIES_WEIGHT = 0.4

# Cross-Encoder Reranking
CROSS_ENCODER_RERANKING_ENABLED = True
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"
CROSS_ENCODER_TOP_N = 8

# 다양성 제약
MAX_CHUNKS_PER_DOC = 3

# ================================
# API 키 (.env에서 로드)
# ================================
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
KMA_API_KEY = os.getenv("KMA_API_KEY", "")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY", "")

# ================================
# GPU 설정 (.env에서 로드)
# ================================
# GPU 할당 전략:
#   - vLLM: GPU 0,1,2,3 (TP=4, run_vllm.sh에서 설정)
#   - RAG Embedding: CPU (vLLM이 GPU 전부 사용)
#   - FAISS: CPU (FAISS_MODE)
#   - Gradio: GPU 없음

CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "")

# RAG/Embedding (빈 값이면 CPU 모드)
RAG_GPU_DEVICE = os.getenv("RAG_GPU_DEVICE", "")

# FAISS 모드: "gpu" 또는 "cpu"
FAISS_MODE = os.getenv("FAISS_MODE", "cpu")

# ================================
# Gradio 설정 (.env에서 로드)
# ================================
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7865"))
GRADIO_THEME = "soft"

# ================================
# 로깅 설정
# ================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ================================
# PDF 파일 설정
# ================================
PDF_FILES = sorted(glob.glob(str(DATA_DIR / "*.pdf")))

# ================================
# 시스템 메시지
# ================================
SYSTEM_MESSAGE = """당신은 재난대응 전문 AI 어시스턴트입니다.

주요 역할:
1. 방대한 재난방재 관련 정보를 분석하고 사용자 요청에 맞는 정보 제공
2. 도시침수 재난 대응 과정에서 실무자의 의사결정을 효과적으로 지원
3. 기존 유사 재난 사례 분석 및 제공을 통한 경험 기반 대응방안 제시
4. 실시간 재난 상황에 적합한 대처방안 추천 및 의사결정 지원
5. 재난방재 관련 규정, 매뉴얼, 표준 운영절차(SOP) 참조하여 근거 있는 정보 제공

답변 시 주의사항:
- 제공된 재난방재 문서 내용을 기반으로 정확하고 실용적인 정보 제공
- 유사 재난 사례와 현재 상황을 비교 분석하여 맞춤형 대응방안 제시
- 불확실한 정보나 추정 내용은 명시적으로 표시하여 의사결정 오류 방지
- 긴급상황 시 신속한 판단을 위한 우선순위 및 단계별 행동지침 제공
- 관련 문서의 해당 부분을 인용하여 신뢰성과 추적가능성 확보
- 한국어로 명확하고 전문적인 어조로 응답

응답 형식:
반드시 답변 시작 부분에 <think> 태그를 사용하여 사고 과정을 기술해야 합니다.

<think>
1. 사용자 질문 분석: 질문의 핵심 의도와 필요한 정보를 파악합니다.
2. 정보 검색 및 검증: 제공된 문서나 검색 결과에서 관련 정보를 찾고 신뢰성을 검증합니다.
3. 대응 방안 수립: 유사 사례와 규정을 바탕으로 최적의 대응책을 논리적으로 구성합니다.
</think>

위의 추론 과정을 바탕으로 <think> 태그가 끝난 후, 구체적이고 실용적인 최종 답변을 제공하세요. 한국어만 사용하세요.

컨텍스트 사용 원칙:
- 기본적으로 현재 사용자 메시지를 기준으로 독립적으로 답변하세요.
- 사용자가 명시적으로 연결을 요청하거나, 현재 질문과 과거 내용의 직접적 연관성이 명확할 때에만 이전 대화/출력을 제한적으로 참조하세요.
"""

# ================================
# 환경 초기화 함수
# ================================
def setup_environment():
    """환경변수 설정"""
    # HuggingFace 캐시 (run_*.sh에서 이미 설정된 경우 유지)
    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = MODEL_CACHE_DIR
    if not os.environ.get("TRANSFORMERS_CACHE"):
        os.environ["TRANSFORMERS_CACHE"] = os.environ.get("HF_HOME", MODEL_CACHE_DIR)

    # CUDA (run_*.sh에서 이미 설정된 경우 유지)
    if CUDA_VISIBLE_DEVICES and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    # SerpAPI
    if SERPAPI_API_KEY:
        os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

    # Gradio 임시 디렉토리
    temp_dir = PROJECT_ROOT.parent / "temp_gradio"
    temp_dir.mkdir(exist_ok=True, mode=0o755)
    os.environ['GRADIO_TEMP_DIR'] = str(temp_dir)
    os.environ['TMPDIR'] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)


def ensure_directories():
    """필요한 디렉토리 생성"""
    for directory in [DATA_DIR, CACHE_DIR, EMBEDDING_CACHE_DIR,
                      VECTORSTORE_CACHE_DIR, OCR_OUTPUT_DIR, TEMP_DIR,
                      Path(MODEL_CACHE_DIR)]:
        directory.mkdir(parents=True, exist_ok=True)


def validate_config():
    """설정 검증"""
    errors = []
    warnings = []

    if not VLLM_SERVER_URL:
        errors.append("VLLM_SERVER_URL이 설정되지 않았습니다.")

    if not MISTRAL_API_KEY:
        warnings.append("MISTRAL_API_KEY가 설정되지 않았습니다. OCR 기능이 제한됩니다.")

    if not SERPAPI_API_KEY:
        warnings.append("SERPAPI_API_KEY가 설정되지 않았습니다. 웹 검색 기능이 제한됩니다.")

    existing_pdfs = [pdf for pdf in PDF_FILES if Path(pdf).exists()]
    if not existing_pdfs:
        warnings.append(f"PDF 파일이 {DATA_DIR}에 없습니다.")

    for w in warnings:
        print(f"⚠️  {w}")

    if errors:
        for e in errors:
            print(f"❌ {e}")
        return False

    return True


# ================================
# 초기화 실행
# ================================
setup_environment()
ensure_directories()


if __name__ == "__main__":
    print(f"🔧 {PROJECT_NAME} {VERSION}")
    print(f"📝 {DESCRIPTION}")
    print(f"🤖 LLM: {LLM_MODEL_NAME}")
    print(f"🌐 vLLM: {VLLM_SERVER_URL}")
    print(f"🧠 임베딩: {EMBEDDING_MODEL}")
    print(f"📁 프로젝트: {PROJECT_ROOT}")
    print(f"📄 PDF: {len([p for p in PDF_FILES if Path(p).exists()])}개")
    print()

    if validate_config():
        print("✅ 설정 검증 완료")
    else:
        print("❌ 설정 검증 실패")
