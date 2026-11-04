#!/usr/bin/env python3
"""Offered load of a workload file, from its contents and arrival spec alone.

For a stable system the long-run rate of LLM calls is lambda * E[calls per
row], whatever the closed loop inside sessions does to their timing, because
every call of an arrived session is eventually released. So the token rates
below are the work the cluster must sustain, not a simulation result.
Utilization rho = lambda * E[S] / replicas needs E[S], the replica-seconds a
row costs, which this script does not model; pass it with --service-sec.

Rows are flat requests (one call) or agent sessions (their sub_requests).
"Uncached input" counts only the tokens a call appends to the previous call's
prompt plus output in the same session: the prefill left if each session's own
history is always a prefix-cache hit. It ignores sharing across sessions.

Usage:
    python analysis/offered_load.py outputs/sessions.jsonl --replicas 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(rows, mean_rate=None, replicas=None, service_sec=None):
    if not rows:
        raise ValueError("workload has no rows")
    arrivals = sorted(row["arrival_time_ns"] for row in rows)
    span = (arrivals[-1] - arrivals[0]) / 1e9
    empirical = (len(rows) - 1) / span if span > 0 else None
    rate = mean_rate if mean_rate is not None else empirical
    totals = dict(calls=0, input_toks=0, uncached_input_toks=0, output_toks=0, tool_delay_sec=0.0)
    for row in rows:
        calls = row.get("sub_requests") or [row]
        previous = 0
        for call in calls:
            totals["calls"] += 1
            totals["input_toks"] += call["input_toks"]
            totals["output_toks"] += call["output_toks"]
            totals["uncached_input_toks"] += max(call["input_toks"] - previous, 0)
            totals["tool_delay_sec"] += call.get("tool_duration_ns", 0) / 1e9
            previous = call["input_toks"] + call["output_toks"]
    per_row = {k: v / len(rows) for k, v in totals.items()}
    out = {"rows": len(rows), "arrival_span_sec": span,
           "spec_mean_rate_per_sec": mean_rate, "empirical_rate_per_sec": empirical,
           "per_row": per_row}
    if rate is None:
        return out
    offered = {"calls_per_sec": rate * per_row["calls"],
               "input_toks_per_sec": rate * per_row["input_toks"],
               "uncached_input_toks_per_sec": rate * per_row["uncached_input_toks"],
               "output_toks_per_sec": rate * per_row["output_toks"],
               # Little's law with each session's tool time as a lower bound
               # on its lifetime.
               "min_concurrent_rows": rate * per_row["tool_delay_sec"]}
    out["offered"] = offered
    if replicas:
        out["offered_per_replica"] = {k: v / replicas for k, v in offered.items()}
    if replicas and service_sec is not None:
        out["utilization"] = rate * service_sec / replicas
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workload", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="arrival manifest; default <workload>.manifest.json when present")
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--service-sec", type=float, dest="service_sec",
                        help="replica-seconds one row costs, E[S], for utilization")
    args = parser.parse_args()
    manifest = args.manifest or args.workload.with_suffix(args.workload.suffix + ".manifest.json")
    mean_rate = None
    if manifest.exists():
        mean_rate = json.loads(manifest.read_text()).get("arrival", {}).get("mean_rate_per_sec")
    with args.workload.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    print(json.dumps(summarize(rows, mean_rate, args.replicas, args.service_sec), indent=2))


if __name__ == "__main__":
    main()
