"""CLI dispatch for workload generators.

Usage:
    python -m workloads.generators sharegpt --model <hf-id> --num-reqs 300 --sps 10 \
        --source <path-or-hf-id> --output workloads/sharegpt-<model>-<n>-sps<r>.jsonl
    python -m workloads.generators swebench --model <hf-id> --num-sessions 50 --sps 0.2 \
        --source <s3-run-or-dir> --output workloads/swe-bench-<model>-<n>-sps<r>.jsonl
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="workloads.generators")
    sub = parser.add_subparsers(dest="generator", required=True)

    sg = sub.add_parser("sharegpt", help="ShareGPT -> LLMServingSim JSONL")
    from workloads.generators.sharegpt import register_args as sg_register
    sg_register(sg)

    sb = sub.add_parser("swebench", help="SWE-bench agent trajectories -> agentic-session JSONL")
    from workloads.generators.swebench import register_args as sb_register
    sb_register(sb)

    args = parser.parse_args()

    if args.generator == "sharegpt":
        from workloads.generators.sharegpt import run
        return run(args)
    if args.generator == "swebench":
        from workloads.generators.swebench import run
        return run(args)

    parser.error(f"Unknown generator: {args.generator}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
