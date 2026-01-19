#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

vllm serve LGAI-EXAONE/EXAONE-4.0-32B \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --dtype float16 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --max-model-len 32768 \
  --host 0.0.0.0 \
  --port 8010