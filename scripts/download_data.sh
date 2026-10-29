#!/usr/bin/env bash
# Benchmarks used by the experiments (not committed: LiveCodeBench v6 is 134 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
[ -f data/HumanEval.jsonl ] || curl -sL https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz | gunzip > data/HumanEval.jsonl
[ -f data/lcb_test6.jsonl ] || curl -sL -o data/lcb_test6.jsonl \
  https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/test6.jsonl
docker pull -q python:3.12-slim
