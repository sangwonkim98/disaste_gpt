"""
Qwen3-32B-AWQ 기반 교직원 업무지원 챗봇 설정
vLLM 서버 + LangChain RAG + Mistral OCR
"""

import os
import glob
import logging
import tempfile
import yaml
from pathlib import Path
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# 설정 로드 함수
# ================================
def load_config():
    """
    설정 로드 우선순위:
    1. 환경변수 (.env 포함)
    2. config.yaml
    3. 코드 내 기본값
    """
    # 1. .env 파일 로드
    load_dotenv()
    
    # 2. config.yaml 파일 로드
    config_path = Path(__file__).parent / "config.yaml"
    yaml_config = {}
    
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f)
            logger.info(f"✅ 설정 파일 로드 완료: {config_path}")
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {e}")
    else:
        logger.warning(f"⚠️ 설정 파일을 찾을 수 없음: {config_path}")

    return yaml_config

# 설정 로드 실행
CONFIG = load_config()

# ================================
# 헬퍼 함수: 설정값 가져오기
# ================================
def get_conf(key_path, default=None, env_key=None):
    """
    YAML 설정과 환경변수를 동시에 확인하여 값을 반환
    - key_path: 'llm.parameters.temperature' 와 같은 점(.)으로 구분된 경로
    - env_key: 'TEMPERATURE' 와 같은 환경변수 키 (오버라이드용)
    """
    # 1. 환경변수 확인
    if env_key and os.getenv(env_key):
        return os.getenv(env_key)
    
    # 2. YAML 설정 확인
    keys = key_path.split('.')
    value = CONFIG
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
            
    return value

# ================================
# 프로젝트 정보
# ================================
PROJECT_NAME = get_conf("project.name", "재난대응 AI 에이전트")
VERSION = get_conf("project.version", "v5.0")
DESCRIPTION = get_conf("project.description", "")

# ================================
# 모델 설정
# ================================
# vLLM 서버 설정
VLLM_SERVER_URL = os.getenv("VLLM_SERVER_URL", "http://0.0.0.0:8010")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

# LLM 설정
LLM_MODEL_NAME = get_conf("llm.model_name", "LGAI-EXAONE/EXAONE-4.0-32B")
ENABLE_REASONING = get_conf("llm.enable_reasoning", True)
# [최적화] Max Tokens를 모델 한계(4096)에 맞춰 2048로 하향 조정
MAX_TOKENS = int(get_conf("llm.max_tokens", 2048))

# 파라미터 (환경변수로도 오버라이드 가능하도록 설정)
TEMPERATURE = float(get_conf("llm.parameters.temperature", 0.6, "TEMPERATURE"))
TOP_P = float(get_conf("llm.parameters.top_p", 0.95, "TOP_P"))
TOP_K = int(get_conf("llm.parameters.top_k", 20, "TOP_K"))
MIN_P = float(get_conf("llm.parameters.min_p", 0, "MIN_P"))

# 임베딩 모델
EMBEDDING_MODEL = get_conf("embedding.model", "dragonkue/BGE-m3-ko")
# 임베딩 디바이스 (GPU 2번 사용 - vLLM은 GPU 0,1 사용)
EMBEDDING_DEVICE = get_conf("embedding.device", "cuda:2", "EMBEDDING_DEVICE")

# ================================
# RAG 설정
# ================================
CHUNK_SIZE = int(get_conf("rag.chunk_size", 5000))
CHUNK_OVERLAP = int(get_conf("rag.chunk_overlap", 3000))
TOP_K_RESULTS = int(get_conf("rag.top_k_results", 8))

# 재순위화(Reranking) 설정
RERANKING_ENABLED = get_conf("rag.reranking.enabled", True)
RERANK_TOP_N = int(get_conf("rag.reranking.top_n", 24))
CROSS_ENCODER_MODEL = get_conf("embedding.cross_encoder_model", "BAAI/bge-reranker-base")
CROSS_ENCODER_TOP_N = int(get_conf("rag.reranking.cross_encoder_top_n", 8))
CROSS_ENCODER_RERANKING_ENABLED = RERANKING_ENABLED # 하위 호환성 유지

# 쿼리 재작성 설정
QUERY_REWRITING_ENABLED = get_conf("rag.query_rewriting.enabled", True)
QUERY_REWRITE_NUM = int(get_conf("rag.query_rewriting.num_rewrites", 3))
QUERY_REWRITE_TIMEOUT = int(get_conf("rag.query_rewriting.timeout", 25))
ORIGINAL_QUERY_WEIGHT = float(get_conf("rag.query_rewriting.original_query_weight", 0.6))
REWRITE_QUERIES_WEIGHT = float(get_conf("rag.query_rewriting.rewrite_queries_weight", 0.4))
CANDIDATES_PER_QUERY = 12 # 기본값 고정

# 다양성 설정
MAX_CHUNKS_PER_DOC = int(get_conf("rag.diversity.max_chunks_per_doc", 3))

# ================================
# API 키 설정
# ================================
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
KMA_API_KEY = os.getenv("KMA_API_KEY", "")

# ================================
# 시스템 메시지
# ================================
SYSTEM_MESSAGE_DEFAULT = """당신은 재난 대응 및 기상 정보를 제공하는 전문 AI 어시스턴트입니다.

[중요: 도구 사용 규칙]
1. **기상청(KMA) 도구 호출 시 절대 좌표나 시간을 직접 계산하지 마십시오.**
   - `nx`, `ny`, `base_date`, `base_time` 파라미터는 시스템이 자동 계산합니다.
   - 당신은 오직 **`location`** (예: '서울', '용인', '부산') 파라미터만 입력하면 됩니다.

2. **상황별 도구 선택 가이드:**
   - "지금 날씨 어때?", "현재 기온은?" → **`kma_get_ultra_srt_ncst`** (초단기실황)
   - "오늘 오후 날씨는?", "내일 비 와?" → **`kma_get_vilage_fcst`** (단기예보)
   - "향후 3시간 뒤 날씨는?" → **`kma_get_ultra_srt_fcst`** (초단기예보)
   - "다음주 날씨는?", "주말 날씨 어때?" → **`kma_get_mid_land_fcst`** (중기예보)
   - "태풍 오고 있어?", "폭염 주의보 있어?" → **`kma_get_wthr_wrn_msg`** (기상특보)
   - "방금 지진 났어?" → **`kma_get_eqk_msg_list`** (지진정보)
   - 그 외 최신 뉴스나 일반 정보 → **`serpapi_web_search`**

3. 답변은 항상 한국어로 명확하고 친절하게 작성하십시오.
"""

SYSTEM_MESSAGE = get_conf("system_prompt", SYSTEM_MESSAGE_DEFAULT)

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

PDF_FILES = sorted(glob.glob(str(DATA_DIR / "*.pdf")))

# ================================
# Gradio 설정
# ================================
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT = int(get_conf("gradio.port", 7865, "GRADIO_PORT"))
GRADIO_THEME = get_conf("gradio.theme", "soft")

# ================================
# 기타 설정
# ================================
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ================================
# 환경 설정 함수
# ================================
def setup_environment():
    import tempfile
    
    # HuggingFace 캐시
    os.environ["HF_HOME"] = MODEL_CACHE_DIR
    os.environ["HUGGINGFACE_CACHE"] = MODEL_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = str(Path(MODEL_CACHE_DIR) / "transformers")
    
    # CUDA
    if CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    
    # SerpAPI (기본값 처리)
    if not os.getenv("SERPAPI_API_KEY"):
        os.environ["SERPAPI_API_KEY"] = "c5ea5598e8e0976c998a5f8e5dce4ab6172d59875a1439ec45ae349d099dd26f"
    
    # 임시 디렉토리
    temp_dir = PROJECT_ROOT.parent / "temp_gradio"
    temp_dir.mkdir(exist_ok=True, mode=0o755)
    temp_dir_str = str(temp_dir)
    
    os.environ['GRADIO_TEMP_DIR'] = temp_dir_str
    os.environ['TMPDIR'] = temp_dir_str
    os.environ['TMP'] = temp_dir_str
    os.environ['TEMP'] = temp_dir_str
    tempfile.tempdir = temp_dir_str
    
    print(f"📁 임시 디렉토리: {temp_dir}")
    print(f"⚙️  설정 파일: hs_code/config.yaml")

def ensure_directories():
    directories = [
        DATA_DIR, CACHE_DIR, EMBEDDING_CACHE_DIR, 
        VECTORSTORE_CACHE_DIR, OCR_OUTPUT_DIR, TEMP_DIR, Path(MODEL_CACHE_DIR)
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_config():
    if not VLLM_SERVER_URL:
        return False
    if not PDF_FILES:
        print(f"❌ PDF 파일이 {DATA_DIR}에 없습니다.")
        return False
    return True

# 초기화 실행
setup_environment()
ensure_directories()

if __name__ == "__main__":
    print(f"🔧 {PROJECT_NAME} {VERSION}")
    print(f"📝 {DESCRIPTION}")
    print(f"🤖 Model: {LLM_MODEL_NAME} (Temp: {TEMPERATURE})")
    print(f"📁 Config loaded from yaml")
