#!/usr/bin/env bash
# Local venv for the AuditBench harness (GPT/vLLM path only). No GPU needed — this
# box only assembles prompts, calls the remote vLLM endpoint, and scores outputs.
# The Gemini/Together clients are stubbed at runtime, so their SDKs are NOT installed.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${VENV:-.venv}"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip
# Minimal deps the upstream GPT path + metrics actually import.
pip install -q "openai>=1.40" "python-dotenv>=1.0" "tenacity>=8.2" tqdm \
  igraph networkx numpy pandas python-dateutil   # upstream helpers + metrics
echo "venv ready: $VENV"
echo "activate with:  source $VENV/bin/activate"
