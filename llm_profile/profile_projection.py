#!/usr/bin/env python3
"""Generate deterministic, explicitly-projected LLMServingSim performance profiles.

The tool never profiles a GPU or allocates model weights.  Its manifests separate
measured H100 calibration inputs from TP=8 and 405B projections.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "llm_profile" / "perf_models"
REQUIRED_LAYERS = {
    "embedding", "input_layernorm", "q_proj", "k_proj", "v_proj", "rope",
    "attn", "o_proj", "post_layernorm", "gate_proj", "up_proj", "act_fn",
    "down_proj", "final_layernorm", "lm_head",
}
DECODE_FIXED_OVERHEAD_NS = 2594
H100_HBM_BYTES_PER_SECOND = 3.35e12
H100_DENSE_BF16_FLOPS_PER_SECOND = 989e12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def layer_index(rows: list[dict[str, str]]) -> dict[tuple[str, int, int], int]:
    return {
        (row["layer_name"], int(row["input"]), int(row["kv_cache"])): int(row["latency(ns)"])
        for row in rows
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    point = (len(ordered) - 1) * fraction
    lo, hi = math.floor(point), math.ceil(point)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (point - lo)


def clamp_speedup(value: float) -> float:
    return min(2.0, max(1.0, value))


def tp_backtest(
    tp1: dict[tuple[str, int, int], int],
    tp2: dict[tuple[str, int, int], int],
    tp4: dict[tuple[str, int, int], int],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    per_layer: dict[str, list[float]] = defaultdict(list)
    report: list[dict[str, Any]] = []
    for key, actual in tp4.items():
        if key not in tp1 or key not in tp2:
            continue
        predicted = tp2[key] / clamp_speedup(tp1[key] / tp2[key])
        ratio = predicted / actual
        per_layer[key[0]].append(ratio)
        report.append({
            "layer_name": key[0], "input": key[1], "kv_cache": key[2],
            "measured_tp4_ns": actual, "predicted_tp4_ns": predicted,
            "signed_error_pct": 100 * (ratio - 1), "absolute_error_pct": 100 * abs(ratio - 1),
        })
    return {layer: statistics.median(values) for layer, values in per_layer.items()}, report


def project_layers_tp8(
    tp1_rows: list[dict[str, str]],
    tp2_rows: list[dict[str, str]],
    tp4_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    tp1, tp2, tp4 = layer_index(tp1_rows), layer_index(tp2_rows), layer_index(tp4_rows)
    bias, backtest = tp_backtest(tp1, tp2, tp4)
    projected: list[dict[str, Any]] = []
    for row in tp4_rows:
        key = (row["layer_name"], int(row["input"]), int(row["kv_cache"]))
        if key not in tp2:
            raise ValueError(f"TP2 profile has no row {key}")
        latency = tp4[key] / clamp_speedup(tp2[key] / tp4[key]) / bias.get(key[0], 1.0)
        projected.append({
            "layer_name": key[0], "input": key[1], "kv_cache": key[2],
            "tp_size": 8, "latency(ns)": max(1, int(round(latency))),
        })
    return projected, backtest, bias


def nearest_attn_speedup(
    tp2: dict[tuple[str, int, int], int],
    tp4: dict[tuple[str, int, int], int],
    chunk: int,
) -> float:
    candidates = sorted(key[1] for key in tp4 if key[0] == "attn" and key[2] == 0 and key in tp2)
    if not candidates:
        return 1.0
    length = min(candidates, key=lambda candidate: abs(candidate - chunk))
    return clamp_speedup(tp2[("attn", length, 0)] / tp4[("attn", length, 0)])


def write_attention_pickles(output: Path, prefill: list[dict[str, Any]], decode: list[dict[str, Any]]) -> None:
    directory = output / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    p = {
        (int(row["kv_cache_size"]), int(row["prefill_chunk_size"])): {
            "kv_cache_size": int(row["kv_cache_size"]),
            "prefill_chunk_size": int(row["prefill_chunk_size"]),
            "latency(ns)": int(row["prediction"]),
        } for row in prefill
    }
    d = {
        (int(row["batch_size"]), int(row["kv_cache_size"])): {
            "batch_size": int(row["batch_size"]),
            "kv_cache_size": int(row["kv_cache_size"]),
            "latency(ns)": int(row["prediction"]),
        } for row in decode
    }
    with (directory / "attn_prefill_prediction_dict.pkl").open("wb") as handle:
        pickle.dump(p, handle, protocol=4)
    with (directory / "attn_decode_prediction_dict.pkl").open("wb") as handle:
        pickle.dump(d, handle, protocol=4)


def project_attention_tp8(
    tp2_rows: list[dict[str, str]], tp4_rows: list[dict[str, str]],
    tp4_dir: Path, output: Path, attn_bias: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tp2, tp4 = layer_index(tp2_rows), layer_index(tp4_rows)
    source_prefill = read_rows(tp4_dir / "predictions" / "attn_prefill_predictions.csv")
    source_decode = read_rows(tp4_dir / "predictions" / "attn_decode_predictions.csv")
    speedups = {
        chunk: nearest_attn_speedup(tp2, tp4, chunk)
        for chunk in {int(row["prefill_chunk_size"]) for row in source_prefill}
    }
    prefill = []
    for row in source_prefill:
        speed = speedups[int(row["prefill_chunk_size"])]
        prefill.append({
            "kv_cache_size": int(row["kv_cache_size"]),
            "prefill_chunk_size": int(row["prefill_chunk_size"]),
            "prediction": max(1, int(round(int(row["prediction"]) / speed / attn_bias))),
        })
    decode = []
    for row in source_decode:
        source = int(row["prediction"])
        latency = DECODE_FIXED_OVERHEAD_NS + max(0, source - DECODE_FIXED_OVERHEAD_NS) / 2
        decode.append({
            "batch_size": int(row["batch_size"]), "kv_cache_size": int(row["kv_cache_size"]),
            "prediction": max(1, int(round(latency / attn_bias))),
        })
    write_rows(output / "predictions" / "attn_prefill_predictions.csv",
               ["kv_cache_size", "prefill_chunk_size", "prediction"], prefill)
    write_rows(output / "predictions" / "attn_decode_predictions.csv",
               ["batch_size", "kv_cache_size", "prediction"], decode)
    write_attention_pickles(output, prefill, decode)
    return prefill, decode


def validate_profile(profile: Path, target_tp: int) -> dict[str, Any]:
    failures: list[str] = []
    rows = read_rows(profile / "layers.csv")
    layers = {row["layer_name"] for row in rows}
    if missing := sorted(REQUIRED_LAYERS - layers):
        failures.append(f"missing layers: {missing}")
    for row in rows:
        if int(row["tp_size"]) != target_tp:
            failures.append(f"incorrect TP: {row}")
            break
        if int(row["latency(ns)"]) <= 0:
            failures.append(f"non-positive latency: {row}")
            break
    for filename, required in (
        ("attn_prefill_predictions.csv", ("kv_cache_size", "prefill_chunk_size", "prediction")),
        ("attn_decode_predictions.csv", ("batch_size", "kv_cache_size", "prediction")),
    ):
        path = profile / "predictions" / filename
        if not path.exists():
            failures.append(f"missing {filename}")
            continue
        for row in read_rows(path):
            if any(key not in row for key in required) or int(row["prediction"]) <= 0:
                failures.append(f"invalid {filename} row: {row}")
                break
    for filename in ("attn_prefill_prediction_dict.pkl", "attn_decode_prediction_dict.pkl"):
        path = profile / "predictions" / filename
        if not path.exists():
            failures.append(f"missing {filename}")
            continue
        with path.open("rb") as handle:
            values = pickle.load(handle)
        if not values or "latency(ns)" not in next(iter(values.values())):
            failures.append(f"invalid ASTRA attention dictionary {filename}")
    return {
        "profile": str(profile.relative_to(REPO)), "target_tp": target_tp,
        "layer_rows": len(rows), "layer_names": sorted(layers),
        "valid": not failures, "failures": failures,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(output: Path, manifest: dict[str, Any]) -> None:
    manifest["generator_git_head"] = git_head()
    manifest["generator_source_sha256"] = sha256(Path(__file__).resolve())
    write_json(output / "profile_manifest.json", manifest)


def project_tp8(args: argparse.Namespace) -> None:
    base = ROOT / args.hardware / args.model
    source = {tp: base / f"tp{tp}" for tp in (1, 2, 4)}
    output = (Path(args.output) if args.output else base / "tp8").resolve()
    tp1, tp2, tp4 = (read_rows(source[tp] / "layers.csv") for tp in (1, 2, 4))
    layers, backtest, bias = project_layers_tp8(tp1, tp2, tp4)
    write_rows(output / "layers.csv", ["layer_name", "input", "kv_cache", "tp_size", "latency(ns)"], layers)
    project_attention_tp8(tp2, tp4, source[4], output, bias.get("attn", 1.0))
    errors = [float(row["absolute_error_pct"]) for row in backtest]
    report = {
        "median_absolute_error_pct": statistics.median(errors),
        "p90_absolute_error_pct": percentile(errors, 0.90),
        "per_layer_bias_ratio": bias, "rows": backtest,
    }
    write_json(output / "tp4_backtest.json", report)
    validation = validate_profile(output, 8)
    write_json(output / "validation.json", validation)
    source_files = {
        str((source[tp] / "layers.csv").relative_to(REPO)): sha256(source[tp] / "layers.csv")
        for tp in (1, 2, 4)
    }
    for filename in ("attn_prefill_predictions.csv", "attn_decode_predictions.csv"):
        path = source[4] / "predictions" / filename
        source_files[str(path.relative_to(REPO))] = sha256(path)
    write_manifest(output, {
        "schema_version": 1, "measurement_status": "projected_not_measured",
        "method": "tp_doubling_ratio_with_tp4_backtest_bias_correction",
        "method_version": args.method_version, "hardware": args.hardware, "model": args.model,
        "precision_bits": 16, "source_tensor_parallel_degrees": [1, 2, 4],
        "target_tensor_parallel_degree": 8, "source_files": source_files,
        "projection_parameters": {
            "speedup_clamp": [1.0, 2.0], "decode_fixed_overhead_ns": DECODE_FIXED_OVERHEAD_NS,
            "attention_bias_ratio": bias.get("attn", 1.0),
        },
        "tp4_backtest": {
            "median_absolute_error_pct": report["median_absolute_error_pct"],
            "p90_absolute_error_pct": report["p90_absolute_error_pct"],
        }, "validation": validation,
    })
    if not validation["valid"]:
        raise SystemExit("profile validation failed")


def load_model(model: str) -> dict[str, Any]:
    return json.loads((REPO / "model_config" / f"{model}.json").read_text())


def operator_metrics(config: dict[str, Any], layer: str, length: int, tp: int, fp_bytes: int = 2) -> tuple[float, float]:
    h, intermediate = int(config["hidden_size"]), int(config["intermediate_size"])
    heads, kv_heads = int(config["num_attention_heads"]), int(config.get("num_key_value_heads", config["num_attention_heads"]))
    head_dim = h // heads
    if layer in {"q_proj", "o_proj"}:
        flops, weights = 2 * length * h * (h // tp), h * (h // tp)
    elif layer in {"k_proj", "v_proj"}:
        flops, weights = 2 * length * h * ((kv_heads * head_dim) // tp), h * ((kv_heads * head_dim) // tp)
    elif layer in {"gate_proj", "up_proj"}:
        flops, weights = 2 * length * h * (intermediate // tp), h * (intermediate // tp)
    elif layer == "down_proj":
        flops, weights = 2 * length * (intermediate // tp) * h, (intermediate // tp) * h
    elif layer == "lm_head":
        flops, weights = 2 * length * h * (int(config["vocab_size"]) // tp), h * (int(config["vocab_size"]) // tp)
    elif layer == "attn":
        flops, weights = 4 * length * length * (heads // tp) * head_dim, 0
    elif layer == "embedding":
        flops, weights = 0, length * h
    else:
        flops, weights = 0, length * h
    activations = max(1, length * h // max(1, tp))
    return float(flops), float((weights + 2 * activations) * fp_bytes)


def is_gemm(layer: str) -> bool:
    return layer in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"}


def project_405_layers(
    source_rows: list[dict[str, str]], source_config: dict[str, Any], target_config: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for row in source_rows:
        layer, length, source_latency = row["layer_name"], int(row["input"]), int(row["latency(ns)"])
        source_flops, source_bytes = operator_metrics(source_config, layer, length, 4)
        target_flops, target_bytes = operator_metrics(target_config, layer, length, 8)
        if is_gemm(layer):
            scale = max(target_flops / max(source_flops, 1), target_bytes / max(source_bytes, 1))
        else:
            scale = target_bytes / max(source_bytes, 1)
        result.append({
            "layer_name": layer, "input": length, "kv_cache": int(row["kv_cache"]),
            "tp_size": 8, "latency(ns)": max(1, int(round(source_latency * scale))),
        })
    return result


def project_405_attention(source_dir: Path, output: Path, factor: float = 1.0) -> None:
    source_pref = read_rows(source_dir / "predictions" / "attn_prefill_predictions.csv")
    source_dec = read_rows(source_dir / "predictions" / "attn_decode_predictions.csv")
    source, target = load_model("meta-llama/Llama-3.1-70B"), load_model("meta-llama/Llama-3.1-405B")
    source_shape = (source["num_attention_heads"] // 4) * (source["hidden_size"] // source["num_attention_heads"])
    target_shape = (target["num_attention_heads"] // 8) * (target["hidden_size"] // target["num_attention_heads"])
    scale = target_shape / source_shape
    prefill = [{"kv_cache_size": int(row["kv_cache_size"]), "prefill_chunk_size": int(row["prefill_chunk_size"]),
                "prediction": max(1, int(round(int(row["prediction"]) * scale * factor)))} for row in source_pref]
    decode = [{"batch_size": int(row["batch_size"]), "kv_cache_size": int(row["kv_cache_size"]),
               "prediction": max(1, int(round(int(row["prediction"]) * scale * factor)))} for row in source_dec]
    write_rows(output / "predictions" / "attn_prefill_predictions.csv",
               ["kv_cache_size", "prefill_chunk_size", "prediction"], prefill)
    write_rows(output / "predictions" / "attn_decode_predictions.csv",
               ["batch_size", "kv_cache_size", "prediction"], decode)
    write_attention_pickles(output, prefill, decode)


def parameter_count(config: dict[str, Any]) -> int:
    h, intermediate = int(config["hidden_size"]), int(config["intermediate_size"])
    kv_dim = int(config["num_key_value_heads"]) * (h // int(config["num_attention_heads"]))
    per_block = 2 * h * h + 2 * h * kv_dim + 3 * h * intermediate + 2 * h
    return int(config["num_hidden_layers"]) * per_block + 2 * int(config["vocab_size"]) * h


def memory_feasibility(config: dict[str, Any], runtime_reserve_gb: float, comm_reserve_gb: float) -> dict[str, Any]:
    tp, pp, fp_bytes, hbm_gb = 8, 2, 2, 80.0
    parameters = parameter_count(config)
    weight_per_gpu = parameters * fp_bytes / (tp * pp)
    kv_per_token = (
        2 * (int(config["num_hidden_layers"]) // pp) * (int(config["num_key_value_heads"]) // tp) *
        (int(config["hidden_size"]) // int(config["num_attention_heads"])) * fp_bytes
    )
    available = hbm_gb * 1e9 - weight_per_gpu - (runtime_reserve_gb + comm_reserve_gb) * 1e9
    return {
        "model": "meta-llama/Llama-3.1-405B", "precision": "bf16",
        "tensor_parallel_degree": tp, "pipeline_parallel_degree": pp, "gpus_per_replica": tp * pp,
        "parameter_count_analytical": parameters, "weight_bytes_per_gpu": weight_per_gpu,
        "kv_bytes_per_token_per_gpu": kv_per_token, "runtime_reserve_gb": runtime_reserve_gb,
        "communication_reserve_gb": comm_reserve_gb, "hbm_capacity_gb": hbm_gb,
        "available_kv_bytes_per_gpu": max(0, available), "max_kv_tokens_per_gpu": max(0, int(available // kv_per_token)),
        "single_128k_context_kv_gb_per_gpu": kv_per_token * 131072 / 1e9,
        "fits_weights_and_reserves": available >= 0,
    }


def physical_limit_check(rows: list[dict[str, Any]], config: dict[str, Any], tp: int) -> dict[str, Any]:
    """Reject a projection that implies more than one H100 can physically supply."""
    peak_flops = 0.0
    peak_hbm_bytes = 0.0
    violations: list[str] = []
    for row in rows:
        flops, hbm_bytes = operator_metrics(config, row["layer_name"], int(row["input"]), tp)
        seconds = int(row["latency(ns)"]) * 1e-9
        peak_flops = max(peak_flops, flops / seconds)
        peak_hbm_bytes = max(peak_hbm_bytes, hbm_bytes / seconds)
    if peak_flops > H100_DENSE_BF16_FLOPS_PER_SECOND:
        violations.append("effective BF16 compute exceeds H100 peak")
    if peak_hbm_bytes > H100_HBM_BYTES_PER_SECOND:
        violations.append("effective HBM traffic exceeds H100 peak")
    return {
        "peak_effective_bf16_flops_per_second": peak_flops,
        "peak_effective_hbm_bytes_per_second": peak_hbm_bytes,
        "valid": not violations,
        "violations": violations,
    }


def project_405b(args: argparse.Namespace) -> None:
    source_dir = ROOT / "H100" / "meta-llama" / "Llama-3.1-70B" / "tp4"
    output = (Path(args.output) if args.output else ROOT / "H100" / "meta-llama" / "Llama-3.1-405B" / "tp8").resolve()
    source, target = load_model("meta-llama/Llama-3.1-70B"), load_model("meta-llama/Llama-3.1-405B")
    nominal = project_405_layers(read_rows(source_dir / "layers.csv"), source, target)
    fields = ["layer_name", "input", "kv_cache", "tp_size", "latency(ns)"]
    write_rows(output / "layers.csv", fields, nominal)
    project_405_attention(source_dir, output)
    tp_base = ROOT / "H100" / "meta-llama" / "Llama-3.1-70B"
    _, backtest_rows, _ = project_layers_tp8(
        read_rows(tp_base / "tp1" / "layers.csv"),
        read_rows(tp_base / "tp2" / "layers.csv"),
        read_rows(tp_base / "tp4" / "layers.csv"),
    )
    uncertainty = percentile([float(row["absolute_error_pct"]) for row in backtest_rows], 0.90) / 100.0
    for name, factor in (("low", max(0.5, 1 - uncertainty)), ("high", 1 + uncertainty)):
        adjusted = [{**row, "latency(ns)": max(1, int(round(int(row["latency(ns)"]) * factor)))} for row in nominal]
        write_rows(output / "variants" / name / "layers.csv", fields, adjusted)
        project_405_attention(source_dir, output / "variants" / name, factor)
    feasibility = memory_feasibility(target, args.runtime_reserve_gb, args.comm_reserve_gb)
    physical_limits = physical_limit_check(nominal, target, 8)
    write_json(output / "memory_feasibility.json", feasibility)
    validation = validate_profile(output, 8)
    write_json(output / "validation.json", validation)
    files = {}
    for path in (source_dir / "layers.csv", source_dir / "predictions" / "attn_prefill_predictions.csv",
                 source_dir / "predictions" / "attn_decode_predictions.csv"):
        files[str(path.relative_to(REPO))] = sha256(path)
    write_manifest(output, {
        "schema_version": 1, "measurement_status": "calibrated_projection_not_measured",
        "method": "operator_geometry_transform_from_measured_h100_70b_tp4",
        "method_version": args.method_version, "hardware": "H100",
        "model": "meta-llama/Llama-3.1-405B", "precision_bits": 16,
        "target_tensor_parallel_degree": 8, "source_files": files,
        "calibration_bounds": {
            "h100_hbm_bytes_per_second": H100_HBM_BYTES_PER_SECOND,
            "h100_dense_bf16_flops_per_second": H100_DENSE_BF16_FLOPS_PER_SECOND,
            "p90_relative_uncertainty": uncertainty,
        }, "memory_feasibility": feasibility, "physical_limit_check": physical_limits,
        "validation": validation,
    })
    if not validation["valid"] or not feasibility["fits_weights_and_reserves"] or not physical_limits["valid"]:
        raise SystemExit("405B profile validation failed")


def compare(args: argparse.Namespace) -> None:
    reference, generated = Path(args.reference), Path(args.generated)
    files = ["layers.csv", "predictions/attn_prefill_predictions.csv", "predictions/attn_decode_predictions.csv"]
    report: dict[str, Any] = {
        "reference": str(reference), "generated": str(generated), "files": {},
        "byte_identical": True, "semantic_identical": True,
    }
    for name in files:
        left, right = reference / name, generated / name
        item = {"reference_sha256": sha256(left), "generated_sha256": sha256(right)}
        item["byte_identical"] = item["reference_sha256"] == item["generated_sha256"]
        if not item["byte_identical"]:
            a, b = read_rows(left), read_rows(right)
            item["reference_rows"], item["generated_rows"] = len(a), len(b)
            item["changed_rows"] = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
            item["semantic_identical"] = item["changed_rows"] == 0
            report["byte_identical"] = False
            report["semantic_identical"] = report["semantic_identical"] and item["semantic_identical"]
        else:
            item["changed_rows"] = 0
            item["semantic_identical"] = True
        report["files"][name] = item
    write_json(Path(args.output), report)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--method-version", default="v1")
    tp8 = commands.add_parser("project-tp8", parents=[common])
    tp8.add_argument("--hardware", default="H100")
    tp8.add_argument("--model", default="meta-llama/Llama-3.1-70B")
    tp8.add_argument("--output")
    tp8.set_defaults(func=project_tp8)
    p405 = commands.add_parser("project-405b", parents=[common])
    p405.add_argument("--output")
    p405.add_argument("--runtime-reserve-gb", type=float, default=8.0)
    p405.add_argument("--comm-reserve-gb", type=float, default=1.0)
    p405.set_defaults(func=project_405b)
    diff = commands.add_parser("compare")
    diff.add_argument("--reference", required=True)
    diff.add_argument("--generated", required=True)
    diff.add_argument("--output", required=True)
    diff.set_defaults(func=compare)
    return root


if __name__ == "__main__":
    options = parser().parse_args()
    options.func(options)
