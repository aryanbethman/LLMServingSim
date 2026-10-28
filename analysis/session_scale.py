#!/usr/bin/env python3
"""Reproducible, sequential agent-session scale experiments on Linux.

prepare copies an existing instance config without changing hardware values.
run records source/input hashes and runs one mode with process-tree monitoring.
compare requires complete, byte-identical request outputs and matching clocks.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from workloads.generators.replicate_sessions import file_sha256, run as replicate

MODES = {
    "legacy": ["--execution-template-mode", "legacy"],
    "legacy-fast": ["--execution-template-mode", "legacy", "--legacy-value-fastpath"],
    "overlay": ["--execution-template-mode", "shared-template", "--template-value-overlay",
                "--template-cache-max-entries", "128"],
    "shared-fast": ["--execution-template-mode", "shared-template", "--template-value-overlay",
                    "--template-cache-max-entries", "128"],
}


def git(*args, cwd=REPO):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def prepare(args):
    config = json.loads(args.base_config.read_text())
    if len(config["nodes"]) != 1:
        raise ValueError("prepare expects a single-node colocated base config")
    node = config["nodes"][0]
    base = node["instances"][0]
    width = base["num_npus"]
    if args.npus <= 0 or args.npus % width:
        raise ValueError("npus must be a positive multiple of the base instance width")
    node["instances"] = [dict(base) for _ in range(args.npus // width)]
    node["num_instances"] = len(node["instances"])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "cluster.json").write_text(json.dumps(config, indent=2) + "\n")
    replicate(argparse.Namespace(source=args.source, output=args.output_dir / "sessions.jsonl",
                                 num_sessions=args.num_sessions, sps=args.sps,
                                 seed=args.seed, first_arrival_sec=0,
                                 token_policy=args.token_policy))
    print(args.output_dir / "cluster.json")


def process_tree_rss(root):
    records = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            records[int(directory.name)] = (int(fields[1]), int(fields[21]))
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root}
    while True:
        found = {pid for pid, (parent, _) in records.items() if parent in descendants}
        if found <= descendants:
            break
        descendants |= found
    return sum(records[pid][1] for pid in descendants if pid in records) * os.sysconf("SC_PAGE_SIZE")


def available_memory():
    return int(re.search(r"^MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text(), re.M)[1]) * 1024


def run(args):
    result = args.result_dir.resolve()
    config = args.config.resolve()
    dataset = args.dataset.resolve()
    lock_name = hashlib.sha256(str(REPO).encode()).hexdigest()[:12]
    with open(f"/tmp/llmss-session-scale-{lock_name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if result.exists():
            raise FileExistsError(f"refusing to overwrite {result}")
        if shutil.disk_usage(REPO).free < args.min_free_gib * 2**30:
            raise RuntimeError("insufficient free disk before launch")
        if available_memory() < args.min_available_gib * 2**30:
            raise RuntimeError("insufficient available memory before launch")
        result.mkdir(parents=True)
        run_id = "session-scale-" + uuid.uuid4().hex[:12]
        cmd = [sys.executable, "-m", "serving", "--cluster-config", str(config),
               "--dataset", os.path.relpath(dataset, REPO),
               "--network-backend", "analytical", "--dtype", "bfloat16", "--block-size", "16",
               "--compact-controller-protocol", "--run-id", run_id,
               "--execution-template-stats-output", str(result / "templates.json"),
               "--load-stats-output", str(result / "load.json"),
               "--load-window-start-sec", str(args.window_start),
               "--load-window-end-sec", str(args.window_end),
               "--output", str(result / "requests.csv"), "--log-level", "WARNING"]
        if args.profile_variant:
            cmd += ["--profile-variant", args.profile_variant]
        cmd += MODES[args.mode]
        env = os.environ.copy()
        env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
        env["LLMSS_VALUE_FASTPATH"] = "0" if args.mode == "overlay" else "1"
        env["LLMSS_VALUE_FASTPATH_ENTRIES"] = "256"
        env["LLMSS_TEMPLATE_BINDINGS_CACHE_BYTES"] = str(16 * 1024 * 1024)
        env["LLMSS_VERIFY_VALUE_FASTPATH"] = "1" if args.verify else "0"
        source_hashes = {str(p.relative_to(REPO)): file_sha256(p)
                         for prefix in ("serving", "analysis", "workloads/generators")
                         for p in (REPO / prefix).rglob("*.py")}
        roots = {"main": REPO, "astra": REPO / "astra-sim",
                 "chakra": REPO / "astra-sim/extern/graph_frontend/chakra"}
        manifest = {"mode": args.mode, "command": cmd, "run_id": run_id,
                    "config_sha256": file_sha256(config), "dataset_sha256": file_sha256(dataset),
                    "source_hashes": source_hashes,
                    "git": {name: {"head": git("rev-parse", "HEAD", cwd=root),
                                    "status": git("status", "--short", cwd=root)}
                            for name, root in roots.items()},
                    "environment": {k: v for k, v in env.items() if k.startswith("LLMSS_")},
                    "limits": {"timeout_seconds": args.timeout, "min_free_gib": args.min_free_gib,
                               "min_available_gib": args.min_available_gib, "max_rss_gib": args.max_rss_gib},
                    "started_unix": time.time()}
        (result / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (result / "environment.txt").open("w") as environment_file:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=environment_file, check=True)
        timed_cmd = ["/usr/bin/time", "-f",
                     '{"wall_seconds":%e,"user_seconds":%U,"system_seconds":%S,"max_rss_kib":%M,"filesystem_output_blocks":%O}',
                     "-o", str(result / "time.json"), *cmd]
        abort = None
        peak_rss = 0
        started = time.monotonic()
        with (result / "stdout.log").open("w") as stdout, (result / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(timed_cmd, cwd=REPO, env=env, stdout=stdout,
                                       stderr=stderr, start_new_session=True)
            with (result / "host.csv").open("w", newline="") as monitor:
                writer = csv.writer(monitor)
                writer.writerow(("wall_seconds", "tree_rss_bytes", "free_disk_bytes", "available_memory_bytes"))
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    rss = process_tree_rss(process.pid)
                    peak_rss = max(peak_rss, rss)
                    free_disk = shutil.disk_usage(REPO).free
                    memory = available_memory()
                    writer.writerow((round(elapsed, 3), rss, free_disk, memory))
                    monitor.flush()
                    if elapsed > args.timeout:
                        abort = "timeout"
                    elif free_disk < args.min_free_gib * 2**30:
                        abort = "disk floor"
                    elif memory < args.min_available_gib * 2**30 or rss > args.max_rss_gib * 2**30:
                        abort = "memory limit"
                    if abort:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        break
                    time.sleep(.5)
        summary = {"mode": args.mode, "exit_code": process.returncode,
                   "aborted": abort, "peak_tree_rss_bytes": peak_rss,
                   "wall_seconds": time.monotonic() - started}
        if (result / "time.json").exists():
            last = (result / "time.json").read_text().strip().splitlines()[-1]
            try:
                summary.update(json.loads(last))
            except json.JSONDecodeError:
                pass
        log = (result / "stdout.log").read_text()
        clock = re.search(r"Total clocks \(ns\):\s+(\d+)", log)
        summary["clock_ns"] = int(clock[1]) if clock else None
        if (result / "requests.csv").exists():
            summary["requests_sha256"] = file_sha256(result / "requests.csv")
            with (result / "requests.csv").open() as handle:
                summary["completed_requests"] = sum(1 for _ in handle) - 1
        if (result / "templates.json").exists():
            stats = json.loads((result / "templates.json").read_text())
            summary["process_cpu_seconds"] = stats.get("process_cpu_seconds")
            summary["graph_cache"] = stats.get("graph_cache")
        if (result / "load.json").exists():
            load = json.loads((result / "load.json").read_text())
            summary["load"] = {k: v for k, v in load.items() if k not in ("bins", "batch_size_histogram")}
        (result / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if process.returncode != 0 or abort:
            # Only this invocation's exact UUID directory, never a shared tree.
            scratch = REPO / "astra-sim/inputs/runs" / run_id
            if scratch.is_dir():
                shutil.rmtree(scratch)
            raise RuntimeError(f"run failed; inspect {result}")


def compare(args):
    rows = [json.loads((path / "summary.json").read_text()) for path in args.results]
    manifests = [json.loads((path / "manifest.json").read_text()) for path in args.results]
    for row in rows:
        if row["exit_code"] != 0 or row["aborted"] or row.get("clock_ns") is None:
            raise ValueError("comparison includes an incomplete run")
    for key in ("config_sha256", "dataset_sha256", "source_hashes"):
        if any(m[key] != manifests[0][key] for m in manifests[1:]):
            raise ValueError(f"comparison inputs differ: {key}")
    for key in ("requests_sha256", "clock_ns", "completed_requests"):
        if any(row[key] != rows[0][key] for row in rows[1:]):
            raise ValueError(f"simulation results differ: {key}")
    print(f"MATCH: {rows[0]['completed_requests']} requests, {rows[0]['clock_ns']} ns")
    for path, row in zip(args.results, rows):
        print(f"{path.name}: {row['wall_seconds']:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--base-config", type=Path, required=True)
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--npus", type=int, required=True)
    prep.add_argument("--num-sessions", type=int, required=True)
    prep.add_argument("--sps", type=float, required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--token-policy", choices=("isolate-copies", "preserve"), default="isolate-copies")
    prep.set_defaults(function=prepare)
    runner = sub.add_parser("run")
    runner.add_argument("--config", type=Path, required=True)
    runner.add_argument("--dataset", type=Path, required=True)
    runner.add_argument("--result-dir", type=Path, required=True)
    runner.add_argument("--mode", choices=MODES, required=True)
    runner.add_argument("--profile-variant", default="bf16-tp4proj")
    runner.add_argument("--window-start", type=float, required=True)
    runner.add_argument("--window-end", type=float, required=True)
    runner.add_argument("--timeout", type=float, default=21600)
    runner.add_argument("--min-free-gib", type=float, default=16)
    runner.add_argument("--min-available-gib", type=float, default=8)
    runner.add_argument("--max-rss-gib", type=float, default=48)
    runner.add_argument("--verify", action="store_true")
    runner.set_defaults(function=run)
    comparison = sub.add_parser("compare")
    comparison.add_argument("results", type=Path, nargs="+")
    comparison.set_defaults(function=compare)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
