#!/usr/bin/env python3
"""Write reproducibility metadata for a persistent scale experiment."""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(repo: Path, *args: str) -> str:
    return subprocess.run(
        args, cwd=repo, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def manifest_for(repo: Path, config: Path, dataset: Path, args: argparse.Namespace) -> dict:
    astra = repo / "astra-sim"
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "logical_npus": args.logical_npus,
        "template_cache_max_entries": args.template_cache_max_entries,
        "model": "meta-llama/Llama-3.1-70B",
        "hardware": "H100",
        "precision_bits": 16,
        "tensor_parallel_degree": 4,
        "workload": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "request_count": 750,
            "arrival_rate": 10,
        },
        "cluster_config": {"path": str(config), "sha256": sha256_file(config)},
        "repository": {
            "head": command(repo, "git", "rev-parse", "HEAD"),
            "status": command(repo, "git", "status", "--short"),
            "working_diff_sha256": hashlib.sha256(
                subprocess.run(
                    ["git", "diff", "--binary"], cwd=repo, check=True,
                    stdout=subprocess.PIPE,
                ).stdout
            ).hexdigest(),
        },
        "astra_sim": {
            "head": command(astra, "git", "rev-parse", "HEAD"),
            "status": command(astra, "git", "status", "--short"),
            "working_diff_sha256": hashlib.sha256(
                subprocess.run(
                    ["git", "diff", "--binary"], cwd=astra, check=True,
                    stdout=subprocess.PIPE,
                ).stdout
            ).hexdigest(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-npus", type=int, required=True)
    parser.add_argument("--template-cache-max-entries", type=int, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest_for(args.repo, args.config, args.dataset, args), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
