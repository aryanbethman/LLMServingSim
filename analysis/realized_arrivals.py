#!/usr/bin/env python3
"""The arrival process and load a simulation actually produced.

A workload fixes only session starts. Inside a session, each later call
arrives when its predecessor completes plus its tool delay, so the call
arrival process is a simulation output. This script joins the simulator's
per-request CSV to the workload it ran and reports:

* closed-loop check: every later call's arrival equals its predecessor's
  completion plus the recorded tool delay;
* the call arrival process: rate, gap CV, count dispersion and gap
  autocorrelation, next to the same numbers for session starts;
* how a session's call-to-call gap splits into queuing, service and tool time;
* carried versus offered call rate (lambda * E[calls per session]);
* load: time-averaged calls in system and in service (by Little's law, the
  sums of their latencies over the window), live sessions, per-replica
  arrivals and imbalance;
* TTFT, queuing delay and latency percentiles.

Request ids follow the workload file: rows in order, a session's calls
consecutive. That holds when the simulator loaded every row.

Usage:
    python analysis/realized_arrivals.py requests.csv sessions.jsonl \
        --window-start 50 --window-end 250
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.arrival_fit import autocorrelation, dispersion

QUANTILES = (0.5, 0.9, 0.99)


def load_requests(path: Path) -> dict[int, dict]:
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out[int(row["request id"])] = {
                "instance": int(row["instance id"]),
                "arrival": int(row["arrival"]),
                "end": int(row["end_time"]),
                "latency": int(row["latency"]),
                "queuing": int(row["queuing_delay"]),
                "ttft": int(row["TTFT"]),
            }
    return out


def load_workload(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def join(requests: dict[int, dict], workload: list[dict]) -> list[list[dict]]:
    """Per workload row, its calls in order, each the CSV record plus the call's tool delay."""
    sessions, next_id = [], 0
    for row in workload:
        calls = row.get("sub_requests") or [row]
        joined = []
        for call in calls:
            if next_id not in requests:
                raise ValueError(f"request {next_id} is missing from the simulator output")
            joined.append(dict(requests[next_id], tool=int(call.get("tool_duration_ns", 0))))
            next_id += 1
        sessions.append(joined)
    if next_id != len(requests):
        raise ValueError(f"simulator output has {len(requests)} requests, workload {next_id}")
    return sessions


def _quantiles(values) -> dict:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {}
    return {"mean": float(values.mean()),
            **{f"p{round(q * 100)}": float(np.quantile(values, q)) for q in QUANTILES}}


def _process(times_sec: np.ndarray) -> dict:
    times_sec = np.sort(times_sec)
    if len(times_sec) < 3:
        return {"arrivals": int(len(times_sec))}
    gaps = np.diff(times_sec)
    span = times_sec[-1] - times_sec[0]
    return {"arrivals": int(len(times_sec)),
            "rate_per_sec": (len(times_sec) - 1) / span if span > 0 else None,
            "gap_cv": float(gaps.std() / gaps.mean()) if gaps.mean() > 0 else None,
            # Windows fine enough that a few-minute measurement window holds 20.
            "count_dispersion": dispersion(times_sec, windows=(1, 2, 5, 10)),
            "gap_autocorrelation": autocorrelation(gaps) if gaps.std() > 0 else None}


def _overlap(begin, end, lo, hi):
    return np.clip(np.minimum(end, hi) - np.maximum(begin, lo), 0, None)


def analyze(sessions: list[list[dict]], window_sec=None, spec_rate=None, replicas=None) -> dict:
    calls = [c for s in sessions for c in s]
    arrival = np.array([c["arrival"] for c in calls], dtype=float)
    end = np.array([c["end"] for c in calls], dtype=float)
    lo, hi = (arrival.min(), end.max()) if window_sec is None else (window_sec[0] * 1e9, window_sec[1] * 1e9)
    if hi <= lo:
        raise ValueError("window end must be after its start")
    width = hi - lo
    inside = (arrival >= lo) & (arrival < hi)

    mismatches, gaps = 0, {"queuing": [], "service": [], "tool": []}
    for session in sessions:
        for prev, call in zip(session, session[1:]):
            mismatches += call["arrival"] != prev["end"] + prev["tool"]
            if lo <= prev["arrival"] < hi:
                gaps["queuing"].append(prev["queuing"])
                gaps["service"].append(prev["latency"] - prev["queuing"])
                gaps["tool"].append(prev["tool"])
    totals = {k: float(np.sum(v)) for k, v in gaps.items()}
    gap_total = sum(totals.values())

    starts = np.array([s[0]["arrival"] for s in sessions], dtype=float)
    session_begin, session_end = starts, np.array([s[-1]["end"] for s in sessions], dtype=float)
    latency = np.array([c["latency"] for c in calls], dtype=float)
    queuing = np.array([c["queuing"] for c in calls], dtype=float)
    instances = np.array([c["instance"] for c in calls])
    calls_per_session = len(calls) / len(sessions)
    replicas = replicas or int(instances.max()) + 1
    per_replica = np.bincount(instances[inside], minlength=replicas)

    report = {
        "window_sec": [lo / 1e9, hi / 1e9],
        "sessions": len(sessions),
        "calls": len(calls),
        "closed_loop_mismatches": int(mismatches),
        "session_starts": _process(starts[(starts >= lo) & (starts < hi)] / 1e9),
        "call_arrivals": _process(arrival[inside] / 1e9),
        "call_gap_share": ({k: v / gap_total for k, v in totals.items()} if gap_total else None),
        "call_gap_sec": {k: _quantiles(np.array(v) / 1e9) for k, v in gaps.items()},
        "carried_calls_per_sec": float(inside.sum() / (width / 1e9)),
        "mean_calls_in_system": float(_overlap(arrival, end, lo, hi).sum() / width),
        "mean_calls_in_service": float(_overlap(arrival + queuing, end, lo, hi).sum() / width),
        "mean_live_sessions": float(_overlap(session_begin, session_end, lo, hi).sum() / width),
        "per_replica_arrivals": per_replica.tolist(),
        "replica_arrival_cv": float(per_replica.std() / per_replica.mean()) if per_replica.mean() else None,
        "ttft_sec": _quantiles(np.array([c["ttft"] for c in calls])[inside] / 1e9),
        "queuing_sec": _quantiles(queuing[inside] / 1e9),
        "latency_sec": _quantiles(latency[inside] / 1e9),
    }
    report["mean_calls_in_service_per_replica"] = report["mean_calls_in_service"] / replicas
    if spec_rate is not None:
        report["spec_session_rate_per_sec"] = spec_rate
        report["offered_calls_per_sec"] = spec_rate * calls_per_session
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("requests", type=Path, help="simulator per-request CSV (--output)")
    parser.add_argument("workload", type=Path, help="the JSONL the simulator ran")
    parser.add_argument("--manifest", type=Path,
                        help="arrival manifest; default <workload>.manifest.json when present")
    parser.add_argument("--window-start", type=float, help="seconds; default the first arrival")
    parser.add_argument("--window-end", type=float, help="seconds; default the last completion")
    parser.add_argument("--replicas", type=int, help="default: highest instance id + 1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.window_start is None) != (args.window_end is None):
        parser.error("give both --window-start and --window-end, or neither")
    window = None if args.window_start is None else (args.window_start, args.window_end)
    manifest = args.manifest or args.workload.with_suffix(args.workload.suffix + ".manifest.json")
    spec_rate = None
    if manifest.exists():
        spec_rate = json.loads(manifest.read_text()).get("arrival", {}).get("mean_rate_per_sec")
    sessions = join(load_requests(args.requests), load_workload(args.workload))
    text = json.dumps(analyze(sessions, window, spec_rate, args.replicas), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
