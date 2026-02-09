#!/bin/bash
# ===========================================
# Gradio 앱 실행 스크립트
# ===========================================
# GPU 할당:
#   - vLLM: GPU 0,1 (run_vllm.sh, GPTQ TP=2)
#   - RAG/App: GPU 2,3 (이 스크립트)
# ===========================================

# 앱 전용 가상환경 활성화
source venv_app/bin/activate

# HuggingFace 캐시 설정 (공용 NAS 사용)
export HF_HOME=/nas/llm/hf
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets

# .env 확인
if [ ! -f ".env" ]; then
    echo "❌ .env 파일이 없습니다. setup.sh를 먼저 실행하세요."
    exit 1
fi

# GPU 격리: vLLM이 GPU 0,1 사용, 앱은 GPU 2,3 사용
export CUDA_VISIBLE_DEVICES=2,3

echo "=========================================="
echo "  재난대응 AI 에이전트 시작"
echo "  RAG/Embedding: GPU 2,3"
echo "=========================================="

python run_server.py
