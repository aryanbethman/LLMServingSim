#!/usr/bin/env python3
"""Generate colocated-replica cluster configs for scalability studies.

One instance per replica, each with its own TP group, filling a target NPU
count. Used to sweep the simulator's own cost -- wall time, memory, and
filesystem traffic -- as NPU count grows, which is what the shared
execution-template path exists to bound.

    python analysis/generate_scale_configs.py --npus 16 72 256 512 1096
    python analysis/generate_scale_configs.py --tp 8 --npus 72

Writes configs/cluster/baseline_scale_<n>.json (or ..._tp<N>_<n>.json when
--tp is not the default 4).
"""

import argparse
import json
from pathlib import Path


MODEL = "meta-llama/Llama-3.1-70B"
HARDWARE = "H100"
CPU_MEMORY = {"mem_size": 1024, "mem_bw": 512, "mem_latency": 0}
NPU_MEMORY = {"mem_size": 80, "mem_bw": 3350, "mem_latency": 0}


def make_config(total_npus: int, tp_size: int, model: str, hardware: str) -> dict:
    """Build one cluster config of `total_npus // tp_size` colocated replicas.

    Called by `main`. Raises when the NPU count is not a whole number of
    replicas, rather than silently rounding and simulating a cluster the
    caller did not ask for.
    """
    if total_npus <= 0 or total_npus % tp_size:
        raise ValueError(
            f"total NPUs ({total_npus}) must be a positive multiple of "
            f"tp_size ({tp_size})"
        )
    replicas = total_npus // tp_size
    instance = {
        "model_name": model,
        "hardware": hardware,
        "npu_mem": NPU_MEMORY,
        "num_npus": tp_size,
        "tp_size": tp_size,
        "pd_type": None,
    }
    return {
        "num_nodes": 1,
        "link_bw": 900,
        "link_latency": 0,
        "nodes": [
            {
                "num_instances": replicas,
                "cpu_mem": CPU_MEMORY,
                "instances": [dict(instance) for _ in range(replicas)],
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="configs/cluster")
    parser.add_argument("--tp", type=int, default=4, dest="tp_size",
                        help="TP degree per replica (default: 4)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--hardware", default=HARDWARE)
    parser.add_argument("--npus", type=int, nargs="+",
                        default=[16, 72, 256, 512, 1096])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for total_npus in args.npus:
        config = make_config(total_npus, args.tp_size, args.model, args.hardware)
        suffix = "" if args.tp_size == 4 else f"tp{args.tp_size}_"
        output = output_dir / f"baseline_scale_{suffix}{total_npus}.json"
        output.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
        print(f"wrote {output}: {total_npus // args.tp_size} "
              f"TP={args.tp_size} replicas")


if __name__ == "__main__":
    main()
