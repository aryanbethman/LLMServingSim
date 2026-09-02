#!/usr/bin/env python3
"""Generate colocated TP=4 replica configurations for scalability studies."""

import argparse
import json
from pathlib import Path


NPU_PER_REPLICA = 4
MODEL = "meta-llama/Llama-3.1-70B"
HARDWARE = "H100"
CPU_MEMORY = {"mem_size": 1024, "mem_bw": 512, "mem_latency": 0}
NPU_MEMORY = {"mem_size": 80, "mem_bw": 3350, "mem_latency": 0}


def make_config(total_npus: int) -> dict:
    if total_npus <= 0 or total_npus % NPU_PER_REPLICA:
        raise ValueError("total NPUs must be a positive multiple of four")
    replicas = total_npus // NPU_PER_REPLICA
    instance = {
        "model_name": MODEL,
        "hardware": HARDWARE,
        "npu_mem": NPU_MEMORY,
        "npu_num": NPU_PER_REPLICA,
        "npu_group": 1,
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
                "instances": [instance.copy() for _ in range(replicas)],
            }
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="cluster_config")
    parser.add_argument("--npus", type=int, nargs="+", default=[16, 72, 256, 512, 1096])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for total_npus in args.npus:
        config = make_config(total_npus)
        output = output_dir / f"baseline_scale_{total_npus}.json"
        output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {output}: {total_npus // NPU_PER_REPLICA} TP=4 replicas")


if __name__ == "__main__":
    main()
