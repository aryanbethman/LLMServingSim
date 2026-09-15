"""Convert a legacy (v0) profile bundle into the current per-category bundle.

The v0 profiler wrote one wide table per bundle:

    <bundle>/layers.csv
        layer_name,input,kv_cache,tp_size,latency(ns)
    <bundle>/predictions/attn_prefill_predictions.csv
        kv_cache_size,prefill_chunk_size,prediction
    <bundle>/predictions/attn_decode_predictions.csv
        batch_size,kv_cache_size,prediction

The simulator now reads a per-category bundle (see the layout comment in
``serving/core/trace_generator.py``):

    <variant>/meta.yaml
    <variant>/tp<N>/dense.csv         layer,tokens,time_us
    <variant>/tp<N>/per_sequence.csv  layer,sequences,time_us
    <variant>/tp<N>/attention.csv     prefill_chunk,kv_prefill,n_decode,kv_decode,time_us

Two things changed besides the file split: operators were fused, and the
unit became microseconds. ``OPERATOR_MAP`` below is the whole of the
first change. The conversion is deterministic and carries no fitting --
every emitted number is a sum, a mean, or a unit conversion of numbers
that are already in the source bundle.

What this cannot recover
------------------------
* ``sampler`` -- the v0 profiler never measured it. Emitted as
  ``--sampler-us`` (default 0.0, which the loader clamps to 1 ns).
  No hardware-specific correction is inferred from another bundle.
* ``skew_fit`` -- the v0 profiler only swept uniform decode batches, so
  there is no alpha to fit. ``meta.yaml`` records ``skew_fit.enabled:
  false`` and ``_skew_alpha`` then falls back to alpha = 0, i.e. no skew
  correction. Converted bundles model uniform-batch attention only and
  are optimistic on heterogeneous batches.

Called by ``profiler/__main__.py`` (the ``export-v0`` subcommand).
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections import defaultdict
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Operator mapping
# ---------------------------------------------------------------------------
#
# v0 profiled vLLM's unfused modules; the current profiler profiles the
# fused ones that vLLM actually executes. Each entry is
# (canonical name) -> (list of v0 names, combine rule).
#
# "sum" is used where the fused kernel does the work of several v0
# kernels back to back. "mean" is used for layernorm only: v0 measured
# input_layernorm and post_layernorm separately and the current trace
# emits one `layernorm` entry twice per block, so the mean preserves the
# per-block total rather than doubling it.

OPERATOR_MAP = {
    "embedding":       (["embedding"], "sum"),
    "layernorm":       (["input_layernorm", "post_layernorm"], "mean"),
    "qkv_proj":        (["q_proj", "k_proj", "v_proj"], "sum"),
    "rotary_emb":      (["rope"], "sum"),
    "o_proj":          (["o_proj"], "sum"),
    "gate_up_proj":    (["gate_proj", "up_proj"], "sum"),
    "act_fn":          (["act_fn"], "sum"),
    "down_proj":       (["down_proj"], "sum"),
    "final_layernorm": (["final_layernorm"], "sum"),
}

# Emitted into per_sequence.csv rather than dense.csv.
PER_SEQUENCE_MAP = {
    "lm_head": (["lm_head"], "sum"),
}

# v0's own quantisation of the attention lookup keys, from
# `_make_attn_db_key` in the v0 simulator: kv rounded up to 64, prefill
# chunk rounded up to 32. The export grid has to land on those
# multiples or the source lookup misses.
KV_GRANULARITY = 64
CHUNK_GRANULARITY = 32

# Default export grid. Doubling on every axis, because the loader
# brackets each axis between its two nearest samples -- a geometric grid
# keeps the relative bracket width constant. Bounded by what v0
# actually swept (kv to 2048); the loader extrapolates linearly above
# the top sample, so queries past the grid degrade rather than fail.
DEFAULT_PREFILL_CHUNKS = [32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_KV = [0, 64, 128, 256, 512, 1024, 2048]
DEFAULT_N_DECODE = [1, 2, 4, 8, 16, 32, 64, 128, 256]


# ---------------------------------------------------------------------------
# Reading the v0 bundle
# ---------------------------------------------------------------------------

def _sha256(path):
    """Content digest of one source file, for meta.yaml provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_layers(path):
    """v0 layers.csv -> {layer_name: {tokens: latency_ns}}.

    Called by `convert_bundle`. v0 wrote one row per (layer, input,
    kv_cache, tp_size); kv_cache is always 0 for the dense layers and
    tp_size is constant within a bundle, so (layer, input) is the key.
    """
    out = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["layer_name"]][int(row["input"])] = int(row["latency(ns)"])
    if not out:
        raise ValueError(f"no rows in {path}")
    return dict(out)


def read_prefill(path):
    """v0 attn_prefill_predictions.csv -> {(kv, chunk): latency_ns}."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["kv_cache_size"]), int(row["prefill_chunk_size"]))
            out[key] = int(float(row["prediction"]))
    if not out:
        raise ValueError(f"no rows in {path}")
    return out


def read_decode(path):
    """v0 attn_decode_predictions.csv -> {(batch, kv): latency_ns}."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["batch_size"]), int(row["kv_cache_size"]))
            out[key] = int(float(row["prediction"]))
    if not out:
        raise ValueError(f"no rows in {path}")
    return out


# ---------------------------------------------------------------------------
# Building the current-format tables
# ---------------------------------------------------------------------------

def _combine(layers, sources, rule, tokens):
    """Apply one OPERATOR_MAP entry at one token count.

    Returns None when any source operator is missing that token count,
    so the caller can skip the row rather than emit a partial sum.
    """
    vals = []
    for name in sources:
        per_token = layers.get(name)
        if per_token is None or tokens not in per_token:
            return None
        vals.append(per_token[tokens])
    total = sum(vals)
    if rule == "mean":
        return total / len(vals)
    return float(total)


def build_dense_rows(layers):
    """{layer: {tokens: ns}} -> dense.csv rows, in microseconds.

    Called by `convert_bundle`. Emits every token count v0 profiled, so
    the dense table is a lossless restatement of the source grid.
    """
    rows = []
    for canonical, (sources, rule) in sorted(OPERATOR_MAP.items()):
        token_grid = sorted(layers.get(sources[0], {}))
        for tokens in token_grid:
            ns = _combine(layers, sources, rule, tokens)
            if ns is not None:
                rows.append((canonical, tokens, ns / 1000.0))
    return rows


def build_per_sequence_rows(layers, sampler_us=0.0):
    """lm_head from v0, plus a `sampler` row v0 never measured.

    The current architecture yaml puts both under `per_sequence`, and
    the loader raises if either is missing, so `sampler` has to be
    emitted even though its honest value is "unknown".
    """
    rows = []
    for canonical, (sources, rule) in sorted(PER_SEQUENCE_MAP.items()):
        for seqs in sorted(layers.get(sources[0], {})):
            ns = _combine(layers, sources, rule, seqs)
            if ns is not None:
                rows.append((canonical, seqs, ns / 1000.0))
    sampler_grid = sorted(layers.get("lm_head", {}))
    for seqs in sampler_grid:
        rows.append(("sampler", seqs, float(sampler_us)))
    return rows


def compose_attention_ns(prefill, decode, prefill_chunk, kv_prefill,
                         n_decode, kv_decode):
    """Reproduce the v0 simulator's attention composition exactly.

    v0 looked the prefill and decode halves up in two separate tables
    and added them (`attn_latency_ns = prefill_attn_latency +
    decode_attn_latency`), keying each half only when that half of the
    batch was non-empty. The current loader wants one 4D table, so the
    addition moves from the simulator into the bundle.

    Raises KeyError when a grid point is not in the source table, which
    is what makes a bad export grid fail loudly instead of silently
    emitting a hole.
    """
    total = 0
    if prefill_chunk > 0:
        total += prefill[(kv_prefill, prefill_chunk)]
    if n_decode > 0:
        total += decode[(n_decode, kv_decode)]
    return total


def build_attention_rows(prefill, decode, chunks=None, kvs=None, n_decodes=None):
    """Full cross product of the four axes, in microseconds.

    Called by `convert_bundle`. The product is emitted whole rather than
    sampled: the loader's bracket assumes every (prefill_chunk,
    n_decode) slice carries the same kv grid, and a full product is the
    cheapest way to guarantee that. Default grid is ~4k rows.
    """
    chunks = DEFAULT_PREFILL_CHUNKS if chunks is None else chunks
    kvs = DEFAULT_KV if kvs is None else kvs
    n_decodes = DEFAULT_N_DECODE if n_decodes is None else n_decodes

    for c in chunks:
        if c % CHUNK_GRANULARITY:
            raise ValueError(f"prefill chunk {c} is not a multiple of "
                             f"{CHUNK_GRANULARITY}; v0 cannot be keyed at it")
    for k in kvs:
        if k % KV_GRANULARITY:
            raise ValueError(f"kv {k} is not a multiple of {KV_GRANULARITY}; "
                             f"v0 cannot be keyed at it")

    rows = []
    for pc in [0] + list(chunks):
        for kvp in kvs:
            for nd in [0] + list(n_decodes):
                for kvd in kvs:
                    if pc == 0 and nd == 0:
                        continue
                    ns = compose_attention_ns(prefill, decode, pc, kvp, nd, kvd)
                    rows.append((pc, kvp, nd, kvd, ns / 1000.0))
    return rows


# ---------------------------------------------------------------------------
# Writing the bundle
# ---------------------------------------------------------------------------

def _write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


def _format_meta(meta):
    """Minimal yaml writer, so the exporter has no import-time dependency
    on PyYAML being present in the profiler environment.
    """
    lines = []

    def emit(obj, indent):
        pad = "  " * indent
        for key, val in obj.items():
            if isinstance(val, dict):
                lines.append(f"{pad}{key}:")
                emit(val, indent + 1)
            elif isinstance(val, list):
                inner = ", ".join(str(v) for v in val)
                lines.append(f"{pad}{key}: [{inner}]")
            elif isinstance(val, bool):
                lines.append(f"{pad}{key}: {'true' if val else 'false'}")
            elif isinstance(val, str):
                lines.append(f"{pad}{key}: {val!r}" if any(
                    c in val for c in ":#") else f"{pad}{key}: {val}")
            else:
                lines.append(f"{pad}{key}: {val}")

    emit(meta, 0)
    return "\n".join(lines) + "\n"


def convert_bundle(src_dir, out_root, hardware, model, variant, tp,
                   sampler_us=0.0, chunks=None, kvs=None, n_decodes=None,
                   provenance=None):
    """Convert one v0 tp<N> bundle into <out_root>/<hw>/<model>/<variant>/tp<N>/.

    Returns (variant_root, counts) where counts is a per-file row count.
    Called by `profiler/__main__.py`; calls `read_*`, `build_*_rows` and
    `_write_csv`.
    """
    layers_csv = os.path.join(src_dir, "layers.csv")
    prefill_csv = os.path.join(src_dir, "predictions", "attn_prefill_predictions.csv")
    decode_csv = os.path.join(src_dir, "predictions", "attn_decode_predictions.csv")
    for path in (layers_csv, prefill_csv, decode_csv):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"v0 bundle is missing {path}")

    layers = read_layers(layers_csv)
    prefill = read_prefill(prefill_csv)
    decode = read_decode(decode_csv)

    dense_rows = build_dense_rows(layers)
    per_seq_rows = build_per_sequence_rows(layers, sampler_us=sampler_us)
    attn_rows = build_attention_rows(prefill, decode, chunks, kvs, n_decodes)

    variant_root = os.path.join(out_root, hardware, model, variant)
    tp_root = os.path.join(variant_root, f"tp{tp}")
    _write_csv(os.path.join(tp_root, "dense.csv"),
               ["layer", "tokens", "time_us"], dense_rows)
    _write_csv(os.path.join(tp_root, "per_sequence.csv"),
               ["layer", "sequences", "time_us"], per_seq_rows)
    _write_csv(os.path.join(tp_root, "attention.csv"),
               ["prefill_chunk", "kv_prefill", "n_decode", "kv_decode", "time_us"],
               attn_rows)
    # Keep the complete two independent surfaces. The compact 4D grid
    # alone cannot preserve source timings between samples or for long
    # prefill chunks. The runtime opts into these self-contained tables
    # only for converted profiles; native profiles retain their 4D lookup.
    _write_csv(os.path.join(tp_root, "attention_prefill_v0.csv"),
               ["kv_cache_size", "prefill_chunk_size", "latency_ns"],
               [(kv, pc, ns) for (kv, pc), ns in sorted(prefill.items())])
    _write_csv(os.path.join(tp_root, "attention_decode_v0.csv"),
               ["batch_size", "kv_cache_size", "latency_ns"],
               [(nd, kv, ns) for (nd, kv), ns in sorted(decode.items())])

    used_chunks = DEFAULT_PREFILL_CHUNKS if chunks is None else list(chunks)
    used_kvs = DEFAULT_KV if kvs is None else list(kvs)
    used_nd = DEFAULT_N_DECODE if n_decodes is None else list(n_decodes)

    meta_path = os.path.join(variant_root, "meta.yaml")
    meta = _merge_meta(meta_path, {
        "profiler_version": "v0-export/1.1.0",
        "hardware": hardware,
        "model": model,
        "variant": variant,
        "architecture": "llama",
        "profiled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "converted from a v0 profile bundle; not a direct measurement",
        "attention_grid": {
            "prefill_chunk": used_chunks,
            "kv": used_kvs,
            "n_decode": used_nd,
            "max_kv": max(used_kvs),
        },
        # No alpha to fit: v0 only swept uniform decode batches. The
        # loader falls back to alpha = 0 (no correction) when this is
        # false, which is the honest reading of an unmeasured axis.
        "skew_fit": {"enabled": False},
        "v0_export": {
            "attention_lookup": "v0-additive",
            "sampler_us": sampler_us,
            "sampler_note": "v0 did not profile the sampler; this value is asserted, not measured",
            "layernorm_rule": "mean(input_layernorm, post_layernorm)",
            "attention_rule": "prefill(kv_prefill, prefill_chunk) + decode(n_decode, kv_decode)",
        },
    }, tp, {
        "layers.csv": _sha256(layers_csv),
        "attn_prefill_predictions.csv": _sha256(prefill_csv),
        "attn_decode_predictions.csv": _sha256(decode_csv),
        "source_dir": src_dir,
        **(provenance or {}),
    })
    with open(meta_path, "w") as f:
        f.write(_format_meta(meta))

    return variant_root, {
        "dense.csv": len(dense_rows),
        "per_sequence.csv": len(per_seq_rows),
        "attention.csv": len(attn_rows),
    }


def _merge_meta(meta_path, base, tp, source_digests):
    """Carry forward an existing meta.yaml's tp list and per-TP digests.

    A variant folder holds several tp<N> folders but only one shared
    meta.yaml, so exporting tp8 after tp4 must not erase tp4's
    provenance. Only the two fields that accumulate are recovered --
    everything else in `base` is regenerated each run.

    Called by `convert_bundle`.
    """
    tp_degrees, sources = [], {}
    if os.path.isfile(meta_path):
        tp_degrees, sources = _read_accumulated_fields(meta_path)

    if tp not in tp_degrees:
        tp_degrees.append(tp)
    sources[f"tp{tp}"] = source_digests

    base["tp_degrees"] = sorted(tp_degrees)
    base["v0_sources"] = {k: sources[k] for k in sorted(sources)}
    return base


def _read_accumulated_fields(meta_path):
    """Pull `tp_degrees` and the `v0_sources` block out of a meta.yaml
    this exporter wrote earlier.

    Deliberately not a general yaml parser: it only has to read back the
    two-level, plain-scalar structure `_format_meta` emits, which keeps
    the exporter free of a PyYAML import. Called by `_merge_meta`.
    """
    tp_degrees, sources = [], {}
    in_sources = False
    current_tp = None
    with open(meta_path) as f:
        for line in f:
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            indent = len(stripped) - len(stripped.lstrip(" "))
            body = stripped.strip()

            if indent == 0:
                in_sources = body.startswith("v0_sources:")
                current_tp = None
                if body.startswith("tp_degrees:"):
                    inner = body.split(":", 1)[1].strip().strip("[]")
                    tp_degrees = [int(v) for v in inner.split(",") if v.strip()]
                continue

            if not in_sources:
                continue
            if indent == 2 and body.endswith(":"):
                current_tp = body[:-1]
                sources[current_tp] = {}
            elif indent >= 4 and current_tp and ":" in body:
                key, val = body.split(":", 1)
                sources[current_tp][key.strip()] = val.strip()
    return tp_degrees, sources
