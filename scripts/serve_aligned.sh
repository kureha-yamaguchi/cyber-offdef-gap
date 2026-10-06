#!/usr/bin/env bash
# Run ON the pod. Installs vLLM (gpt-oss capable) and serves the aligned model.
set -euo pipefail
MODEL="${MODEL:-openai/gpt-oss-120b}"
pip install -q -U "vllm>=0.10.1" hf_transfer 2>&1 | tail -5 || true
export HF_HUB_ENABLE_HF_TRANSFER=1
nohup vllm serve "$MODEL" \
  --port 8000 --host 0.0.0.0 \
  --async-scheduling \
  > /workspace/vllm.log 2>&1 &
echo "vllm serving $MODEL (pid $!); tail -f /workspace/vllm.log"
