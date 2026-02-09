#!/bin/bash
# vLLM 전용 가상환경 활성화
source venv_vllm/bin/activate

# 캐시 설정
export HF_HOME=/nas/user77/workspace/Project/disaster_gpt/hs_code/cache/models
export CUDA_VISIBLE_DEVICES=0,1

echo "=========================================="
echo "  vLLM 서버 시작 (GPTQ 양자화)"
echo "  GPU: 0,1 (TP=2)"
echo "=========================================="

vllm serve LGAI-EXAONE/EXAONE-4.0-32B-GPTQ \
  --quantization gptq \
  --dtype auto \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --max-num-seqs 4 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser deepseek_r1 \
  --port 8000 \
  --host 0.0.0.0