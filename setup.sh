#!/bin/bash
# ===========================================
# 환경 설정 스크립트
# vLLM 서버용 / 앱용 가상환경 분리
# Python 3.10 사용 (tokenizers wheel 호환)
# ===========================================

set -e

# Python 3.10 경로
PYTHON="/usr/bin/python3.10"

# Python 버전 확인
if [ ! -f "$PYTHON" ]; then
    echo "❌ Python 3.10이 없습니다. 설치가 필요합니다."
    exit 1
fi

echo "=========================================="
echo "  재난대응 AI 에이전트 환경 설정"
echo "  Python: $($PYTHON --version)"
echo "=========================================="
echo ""

# -----------------------------
# 기존 가상환경 정리
# -----------------------------
for dir in venv venv_vllm venv_app hs_venv; do
    if [ -d "$dir" ]; then
        echo "기존 $dir 삭제..."
        rm -rf "$dir"
    fi
done

# -----------------------------
# 1. vLLM 서버용 가상환경
# -----------------------------
echo ""
echo "[1/2] vLLM 서버용 가상환경 설정..."

echo "  - venv_vllm 생성..."
$PYTHON -m venv venv_vllm

echo "  - 패키지 설치..."
source venv_vllm/bin/activate
pip install --upgrade pip -q
pip install -r requirements-vllm.txt
deactivate

echo "  ✅ vLLM 환경 완료"

# -----------------------------
# 2. 앱용 가상환경
# -----------------------------
echo ""
echo "[2/2] 앱용 가상환경 설정..."

echo "  - venv_app 생성..."
$PYTHON -m venv venv_app

echo "  - 패키지 설치..."
source venv_app/bin/activate
pip install --upgrade pip -q
pip install -r requirements-app.txt
deactivate

echo "  ✅ 앱 환경 완료"

# -----------------------------
# 3. .env 파일 확인
# -----------------------------
echo ""
if [ ! -f ".env" ]; then
    echo "⚠️  .env 파일이 없습니다. .env.example을 복사합니다..."
    cp .env.example .env
    echo "   .env 파일을 열어 API 키를 설정하세요."
fi

echo ""
echo "=========================================="
echo "  설치 완료!"
echo "=========================================="
echo ""
echo "📁 생성된 환경:"
echo "  - venv_vllm/  : vLLM 서버용"
echo "  - venv_app/   : Gradio 앱용"
echo ""
echo "🚀 사용법:"
echo "  1. vLLM 서버:  ./run_vllm.sh"
echo "  2. 앱 실행:    ./run_app.sh"
echo ""
