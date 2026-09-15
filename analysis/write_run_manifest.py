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


def git_metadata(repo: Path) -> dict:
    return {
        "head": command(repo, "git", "rev-parse", "HEAD"),
        "status": command(repo, "git", "status", "--short"),
        "working_diff_sha256": hashlib.sha256(
            subprocess.run(
                ["git", "diff", "--binary"], cwd=repo, check=True,
                stdout=subprocess.PIPE,
            ).stdout
        ).hexdigest(),
    }


def untracked_source_hashes(repo: Path) -> dict[str, str]:
    prefixes = ("analysis/", "configs/", "experiments/", "profiler/", "serving/", "tests/")
    skipped = ("bigworkloads/", "__pycache__/", "/logs/", "/outputs/")
    suffixes = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
    paths = command(repo, "git", "ls-files", "--others", "--exclude-standard").splitlines()
    result = {}
    for relative in paths:
        path = repo / relative
        if (not relative.startswith(prefixes) or path.suffix not in suffixes
                or any(token in relative for token in skipped) or not path.is_file()
                or path.stat().st_size > 10 * 1024 * 1024):
            continue
        result[relative] = sha256_file(path)
    return result


def config_profiles(config: dict, logical_npus: int) -> list[dict]:
    profiles = []
    for node in config.get("nodes", []):
        for instance in node.get("instances", []):
            profiles.append({
                "model": instance.get("model_name"),
                "hardware": instance.get("hardware"),
                "tensor_parallel_degree": instance.get("tp_size"),
                "pipeline_parallel_degree": instance.get("pp_size", 1),
                "num_npus": instance.get("num_npus"),
            })
    if not profiles or any(any(p[key] is None for key in (
            "model", "hardware", "tensor_parallel_degree", "num_npus"))
            for p in profiles):
        raise ValueError("scale manifest requires explicit model, hardware, tp_size and num_npus")
    total_npus = sum(profile["num_npus"] for profile in profiles)
    if total_npus != logical_npus:
        raise ValueError(
            f"logical_npus={logical_npus} does not match cluster config NPU count={total_npus}"
        )
    return profiles


def uniform_or_list(profiles: list[dict], key: str):
    values = sorted({profile[key] for profile in profiles})
    if len(values) == 1:
        return values[0]
    return values


def profile_bundle_hashes(repo: Path, profiles: list[dict], variant: str) -> dict[str, str]:
    """Record the profile content of the variant this scale run selected."""
    result = {}
    for profile in profiles:
        root = repo / "profiler/perf" / profile["hardware"] / profile["model"] / variant
        candidates = [root / "meta.yaml"]
        candidates.extend((root / f'tp{profile["tensor_parallel_degree"]}').glob("*.csv"))
        for path in candidates:
            if path.is_file():
                result[str(path.relative_to(repo))] = sha256_file(path)
    return result


def manifest_for(repo: Path, config: Path, dataset: Path, args: argparse.Namespace) -> dict:
    astra = repo / "astra-sim"
    cluster = json.loads(config.read_text())
    profiles = config_profiles(cluster, args.logical_npus)
    astra_metadata = git_metadata(astra)
    chakra = astra / "extern/graph_frontend/chakra"
    chakra_metadata = git_metadata(chakra)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "logical_npus": args.logical_npus,
        "template_cache_max_entries": args.template_cache_max_entries,
        "model": uniform_or_list(profiles, "model"),
        "hardware": uniform_or_list(profiles, "hardware"),
        "precision_bits": 16,
        "tensor_parallel_degree": uniform_or_list(profiles, "tensor_parallel_degree"),
        "pipeline_parallel_degree": uniform_or_list(profiles, "pipeline_parallel_degree"),
        "cluster_profiles": profiles,
        "workload": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "request_count": 750,
            "arrival_rate": 10,
        },
        "cluster_config": {"path": str(config), "sha256": sha256_file(config)},
        "repository": git_metadata(repo),
        "untracked_source_hashes": untracked_source_hashes(repo),
        "profile_variant": args.profile_variant,
        "profile_bundle_hashes": profile_bundle_hashes(repo, profiles, args.profile_variant),
        "astra_sim": {**astra_metadata, "chakra": chakra_metadata},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logical-npus", type=int, required=True)
    parser.add_argument("--template-cache-max-entries", type=int, required=True)
    parser.add_argument("--profile-variant", default="bf16")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest_for(args.repo, args.config, args.dataset, args), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
