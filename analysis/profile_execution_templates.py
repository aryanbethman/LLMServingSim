#!/usr/bin/env python3
"""Profile exact and structural reuse among rank-specific Chakra ET files."""

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter


RANK_ATTRIBUTES = {"comm_src", "comm_dst", "comm_tag"}
RANK_NAME = re.compile(r"^(COMM_(?:SEND|RECV)_NODE_.*)_\\d+_\\d+$")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_chakra(repo_root):
    chakra_root = repo_root / "astra-sim" / "extern" / "graph_frontend"
    if str(chakra_root) not in sys.path:
        sys.path.insert(0, str(chakra_root))
    from chakra.schema.protobuf.et_def_pb2 import GlobalMetadata, Node
    from chakra.src.third_party.utils.protolib import decodeMessage
    return GlobalMetadata, Node, decodeMessage


def structural_digest(path, node_type, metadata_type, decode_message):
    """Hash ET graph structure while removing rank-specific point-to-point fields."""
    digest = hashlib.sha256()
    nodes = 0
    with path.open("rb") as handle:
        metadata = metadata_type()
        if not decode_message(handle, metadata):
            raise ValueError(f"Missing Chakra metadata: {path}")
        while True:
            node = node_type()
            if not decode_message(handle, node):
                break
            normalized = node_type()
            normalized.CopyFrom(node)
            match = RANK_NAME.match(normalized.name)
            if match:
                normalized.name = match.group(1) + "_<src>_<dst>"
            attrs = [attr for attr in normalized.attr if attr.name not in RANK_ATTRIBUTES]
            del normalized.attr[:]
            normalized.attr.extend(attrs)
            payload = normalized.SerializeToString(deterministic=True)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            nodes += 1
    return digest.hexdigest(), nodes


def et_batches(workload_root):
    for directory, _, filenames in os.walk(workload_root):
        et_files = sorted(
            Path(directory) / filename
            for filename in filenames
            if filename.startswith("llm.") and filename.endswith(".et")
        )
        if et_files:
            yield Path(directory), et_files


def percentage_saved(total, distinct):
    return 0.0 if not total else round((1.0 - distinct / total) * 100, 3)


def main():
    parser = argparse.ArgumentParser(
        description="Profile content-addressed template opportunities in Chakra ET output."
    )
    parser.add_argument("--workload-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="limit processed batch directories; 0 profiles all")
    parser.add_argument("--raw-only", action="store_true",
                        help="skip protobuf parsing and structural fingerprints")
    args = parser.parse_args()

    workload_root = args.workload_root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    if not workload_root.is_dir():
        parser.error(f"workload root does not exist: {workload_root}")

    metadata_type = node_type = decode_message = None
    if not args.raw_only:
        metadata_type, node_type, decode_message = load_chakra(repo_root)

    raw_hashes = Counter()
    structural_hashes = Counter()
    raw_sizes = {}
    structural_sizes = {}
    rank_counts = Counter()
    node_counts = []
    parse_failures = []
    batch_count = 0
    started = perf_counter()

    for batch_dir, et_files in et_batches(workload_root):
        if args.max_batches and batch_count >= args.max_batches:
            break
        batch_count += 1
        rank_counts[len(et_files)] += 1
        for et_file in et_files:
            size = et_file.stat().st_size
            raw_hash = sha256_file(et_file)
            raw_hashes[raw_hash] += 1
            raw_sizes.setdefault(raw_hash, size)
            if args.raw_only:
                continue
            try:
                structural_hash, node_count = structural_digest(
                    et_file, node_type, metadata_type, decode_message
                )
            except Exception as error:  # surface malformed traces without aborting profile
                parse_failures.append({"path": str(et_file), "error": str(error)})
                continue
            structural_hashes[structural_hash] += 1
            structural_sizes.setdefault(structural_hash, size)
            node_counts.append(node_count)

    total_files = sum(raw_hashes.values())
    total_bytes = sum(size * raw_hashes[digest] for digest, size in raw_sizes.items())
    exact_template_bytes = sum(raw_sizes.values())
    report = {
        "workload_root": str(workload_root),
        "batches_profiled": batch_count,
        "rank_files": total_files,
        "rank_files_per_batch": dict(sorted(rank_counts.items())),
        "rank_et_bytes": total_bytes,
        "exact_templates": len(raw_hashes),
        "exact_template_bytes": exact_template_bytes,
        "exact_file_reuse_percent": percentage_saved(total_files, len(raw_hashes)),
        "exact_byte_reuse_percent": percentage_saved(total_bytes, exact_template_bytes),
        "structural_templates": len(structural_hashes) if not args.raw_only else None,
        "structural_template_bytes_upper_bound": (
            sum(structural_sizes.values()) if not args.raw_only else None
        ),
        "structural_file_reuse_percent": (
            percentage_saved(total_files, len(structural_hashes)) if not args.raw_only else None
        ),
        "structural_byte_reuse_upper_bound_percent": (
            percentage_saved(total_bytes, sum(structural_sizes.values()))
            if not args.raw_only else None
        ),
        "nodes_per_rank_et": {
            "min": min(node_counts) if node_counts else None,
            "max": max(node_counts) if node_counts else None,
            "mean": round(sum(node_counts) / len(node_counts), 3) if node_counts else None,
        },
        "parse_failures": parse_failures,
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
