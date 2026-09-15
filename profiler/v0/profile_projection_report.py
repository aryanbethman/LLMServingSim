#!/usr/bin/env python3
"""Emit compact validation tables and a plot for a projected TP=8 profile."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    lo, hi = int(position), min(len(values) - 1, int(position) + 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.profile / "tp4_backtest.json").read_text())
    rows = report["rows"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["layer_name"]].append(float(row["absolute_error_pct"]))
    with (args.output_dir / "tp4_backtest_by_layer.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["layer_name", "rows", "median_ape_pct", "p90_ape_pct"])
        writer.writeheader()
        for layer, errors in sorted(grouped.items()):
            writer.writerow({"layer_name": layer, "rows": len(errors),
                             "median_ape_pct": f"{statistics.median(errors):.4f}",
                             "p90_ape_pct": f"{percentile(errors, .9):.4f}"})

    measured = [float(row["measured_tp4_ns"]) for row in rows]
    predicted = [float(row["predicted_tp4_ns"]) for row in rows]
    errors = [float(row["absolute_error_pct"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].scatter(measured, predicted, s=4, alpha=.35, color="#147d64")
    lower, upper = min(measured + predicted), max(measured + predicted)
    axes[0].plot([lower, upper], [lower, upper], "--", color="#555555", linewidth=1)
    axes[0].set(xscale="log", yscale="log", xlabel="Measured TP=4 latency (ns)",
                ylabel="Predicted TP=4 latency (ns)", title="Back-test: TP1/2 → TP4")
    axes[1].hist(errors, bins=40, color="#147d64", edgecolor="white")
    axes[1].axvline(statistics.median(errors), color="#222222", linestyle="--", label="median")
    axes[1].axvline(percentile(errors, .9), color="#be5b32", linestyle="--", label="P90")
    axes[1].set(xlabel="Absolute percentage error", ylabel="Rows", title="Back-test error")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "tp4_backtest.png", dpi=180)
    plt.close(fig)

    summary = {"median_absolute_error_pct": statistics.median(errors),
               "p90_absolute_error_pct": percentile(errors, .9), "rows": len(rows),
               "interpretation": "Measured H100 TP=4 is held out; this validates the projection method, not TP=8 hardware."}
    (args.output_dir / "tp4_backtest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
