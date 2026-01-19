#!/bin/bash

# 스크립트가 있는 디렉토리로 이동
cd "$(dirname "$0")"

# 가상환경 활성화
source hs_venv/bin/activate

# 필요한 패키지 확인 (PyYAML 등)
pip install -q pyyaml python-dotenv gradio langchain-community

# Python 경로 설정 (현재 디렉토리를 모듈 경로에 추가)
export PYTHONPATH=$PYTHONPATH:$(pwd)
# vLLM이 GPU를 사용하므로, 애플리케이션 로직은 충돌 방지를 위해 CPU 모드로 실행 (config.yaml device: cpu 설정 준수)
export CUDA_VISIBLE_DEVICES=""

# 애플리케이션 실행
echo "🚀 Starting Application (LangGraph Mode)..."
python hs_code/main_graph.py "$@" > run.log 2>&1 &
tail -f run.log
