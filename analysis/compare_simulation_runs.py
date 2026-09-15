#!/usr/bin/env python3
"""Compare deterministic LLMServingSim request outputs and simulated metrics."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
METRIC_LABELS = (
    "Total requests",
    "Total clocks (ns)",
    "Total latency (s)",
    "Total input tokens",
    "Total generated tokens",
    "Request throughput (req/s)",
    "Average prompt throughput (tok/s)",
    "Average generation throughput (tok/s)",
    "Total token throughput (tok/s)",
)


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        required = {"instance id", "request id"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing identifier columns: {sorted(missing)}")
        rows = {}
        for row in reader:
            key = (row["instance id"], row["request id"])
            if key in rows:
                raise ValueError(f"{path}: duplicate request row {key}")
            rows[key] = row
    return tuple(reader.fieldnames), rows


def load_metrics(path: Path):
    text = ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    metrics = {}
    for label in METRIC_LABELS:
        match = re.search(r"^" + re.escape(label) + r":\s*(.+?)\s*$", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"{path}: could not find metric {label!r}")
        metrics[label] = match.group(1)
    return metrics


def compare_rows(reference, candidate):
    ref_header, ref_rows = reference
    cand_header, cand_rows = candidate
    differences = []
    if ref_header != cand_header:
        differences.append("CSV headers differ")
        return differences
    if set(ref_rows) != set(cand_rows):
        only_ref = sorted(set(ref_rows) - set(cand_rows))
        only_cand = sorted(set(cand_rows) - set(ref_rows))
        differences.append(
            f"request identifiers differ: only reference={only_ref[:5]}, "
            f"only candidate={only_cand[:5]}"
        )
        return differences
    for key in sorted(ref_rows):
        for field in ref_header:
            if ref_rows[key][field] != cand_rows[key][field]:
                differences.append(
                    f"request {key}, field {field!r}: "
                    f"{ref_rows[key][field]!r} != {cand_rows[key][field]!r}"
                )
                if len(differences) == 20:
                    return differences
    return differences


def main():
    parser = argparse.ArgumentParser(
        description="Compare retained-trace and cleanup-enabled LLMServingSim runs."
    )
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    args = parser.parse_args()

    differences = compare_rows(load_rows(args.reference_csv), load_rows(args.candidate_csv))
    ref_metrics = load_metrics(args.reference_log)
    candidate_metrics = load_metrics(args.candidate_log)
    for label in METRIC_LABELS:
        if ref_metrics[label] != candidate_metrics[label]:
            differences.append(
                f"metric {label!r}: {ref_metrics[label]!r} != "
                f"{candidate_metrics[label]!r}"
            )

    if differences:
        print("SIMULATION OUTPUTS DIFFER", file=sys.stderr)
        for difference in differences:
            print("- " + difference, file=sys.stderr)
        raise SystemExit(1)

    print("SIMULATION OUTPUTS MATCH")
    print(f"Requests compared: {len(load_rows(args.reference_csv)[1])}")
    for label in METRIC_LABELS:
        print(f"{label}: {ref_metrics[label]}")


if __name__ == "__main__":
    main()
