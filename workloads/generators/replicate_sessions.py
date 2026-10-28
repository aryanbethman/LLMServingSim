"""Reschedule complete recorded sessions without changing their call chains.

Token IDs are simulator hash inputs, not model inputs. ``isolate-copies``
relabels each repeated source corpus into a disjoint integer range. It keeps
all prefix equalities within a copy, including cross-task system prefixes,
without introducing cache reuse between independent copies of the corpus.
``preserve`` explicitly models repeated identical content instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def register_args(parser):
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-sessions", type=int, required=True)
    parser.add_argument("--sps", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--first-arrival-sec", type=float, default=0)
    parser.add_argument("--token-policy", choices=("isolate-copies", "preserve"),
                        default="isolate-copies")


def load_sessions(path):
    sessions = []
    identifiers = set()
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            row = json.loads(line)
            sid = row.get("session_id")
            if not isinstance(sid, str) or sid in identifiers:
                raise ValueError(f"line {number}: missing or duplicate session_id")
            identifiers.add(sid)
            calls = row.get("sub_requests")
            if not isinstance(calls, list) or not calls:
                raise ValueError(f"line {number}: expected a nonempty session")
            for call in calls:
                for field in ("input_toks", "output_toks", "tool_duration_ns"):
                    value = call.get(field, 0 if field == "tool_duration_ns" else None)
                    if type(value) is not int or value < 0:
                        raise ValueError(f"{sid}: invalid {field}")
                for field, count in (("input_tok_ids", "input_toks"),
                                     ("output_tok_ids", "output_toks")):
                    if field in call:
                        ids = call[field]
                        if len(ids) != call[count] or any(type(x) is not int or x < 0 for x in ids):
                            raise ValueError(f"{sid}: invalid {field}")
            sessions.append(row)
    if not sessions:
        raise ValueError("source contains no sessions")
    return sessions


def replicated_session(source, index, copy_index, arrival_ns, stride, policy):
    offset = copy_index * stride if policy == "isolate-copies" else 0
    calls = []
    for original in source["sub_requests"]:
        call = dict(original)
        for field in ("input_tok_ids", "output_tok_ids"):
            if field in call and offset:
                call[field] = [token + offset for token in call[field]]
        calls.append(call)
    return dict(source, session_id=f"replica-{index:08d}:{source['session_id']}",
                arrival_time_ns=arrival_ns, sub_requests=calls)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    if args.num_sessions <= 0 or not math.isfinite(args.sps) or args.sps <= 0:
        raise ValueError("num-sessions and sps must be positive")
    if not math.isfinite(args.first_arrival_sec) or args.first_arrival_sec < 0:
        raise ValueError("first-arrival-sec must be finite and nonnegative")
    if args.output.resolve() == args.source.resolve():
        raise ValueError("output must differ from source")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite a workload or manifest")
    sessions = load_sessions(args.source)
    stride = 1 + max((token for row in sessions for call in row["sub_requests"]
                      for field in ("input_tok_ids", "output_tok_ids")
                      for token in call.get(field, [])), default=0)
    order_rng = random.Random(args.seed)
    arrival_rng = random.Random(args.seed + 1)
    arrival = args.first_arrival_sec * 1e9
    args.output.parent.mkdir(parents=True, exist_ok=True)
    provenance = []
    digest = hashlib.sha256()
    total_calls = 0
    with args.output.open("x") as handle:
        index = 0
        while index < args.num_sessions:
            order = list(range(len(sessions)))
            order_rng.shuffle(order)
            copy_index = index // len(sessions)
            for source_index in order:
                if index >= args.num_sessions:
                    break
                arrival += arrival_rng.expovariate(args.sps) * 1e9
                row = replicated_session(sessions[source_index], index, copy_index,
                                         round(arrival), stride, args.token_policy)
                encoded = json.dumps(row, separators=(",", ":")) + "\n"
                handle.write(encoded)
                digest.update(encoded.encode())
                total_calls += len(row["sub_requests"])
                provenance.append({"session_id": row["session_id"],
                                   "source_session_id": sessions[source_index]["session_id"],
                                   "copy_index": copy_index,
                                   "arrival_time_ns": row["arrival_time_ns"]})
                index += 1
    manifest = {"schema_version": 1, "source": str(args.source),
                "source_sha256": file_sha256(args.source),
                "output_sha256": digest.hexdigest(), "seed": args.seed, "sps": args.sps,
                "num_sessions": args.num_sessions, "num_calls": total_calls,
                "token_policy": args.token_policy, "token_namespace_stride": stride,
                "first_arrival_sec": args.first_arrival_sec,
                "fidelity": "Complete call chains, lengths, tool delays and within-copy prefix equality preserved; only IDs and session arrivals change.",
                "sessions": provenance}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.num_sessions} complete sessions / {total_calls} calls to {args.output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    register_args(parser)
    run(parser.parse_args())
