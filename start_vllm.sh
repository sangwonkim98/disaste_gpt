#!/bin/bash
source hs_venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
  --quantization awq \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.5 \
  --dtype auto \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8010