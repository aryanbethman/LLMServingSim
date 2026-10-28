#!/bin/bash
# Generate swe-bench-qwen3-coder-30b-a3b-50-sps0.2.jsonl from the Qwen team's
# OpenHands + Qwen3-Coder-30B-A3B-Instruct run on SWE-bench Verified.
#
# No GPU needed: token counts and tool durations come from the recorded run,
# and the model is only used for its tokenizer and chat template. The first
# run downloads ~370 MB of trajectories into ~/.cache/llmservingsim/swebench.
# Simulate the output with a Qwen3-30B-A3B config (same architecture), e.g.
# configs/cluster/single_node_moe_single_instance.json.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"

python3 -m workloads.generators swebench \
    --model "$MODEL" \
    --num-sessions 50 --sps 0.2 --seed 42 \
    --output workloads/swe-bench-qwen3-coder-30b-a3b-50-sps0.2.jsonl

    # Cap session length or context to keep the file and the run small
    # --max-calls 40 \
    # --max-kv-toks 32768
