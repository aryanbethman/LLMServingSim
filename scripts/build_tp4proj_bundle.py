"""Build perf/<hw>/<model>/bf16-tp4proj in the vllm-sim fork's profile format.

The fork's own bf16 bundle holds measured tp1 and tp2. The a4053bc bf16-tp4proj
bundle is in an older format the fork cannot load (legacy skew fits, different
CSV columns), so this redoes its projection recipe on the fork's data:
  tp1, tp2     copied unchanged from bf16
  tp4          dense / per_sequence / attention: tp2 time / clamp(tp1 time / tp2 time, 1, 2), per row
               skew.csv and its metadata copied from tp2; skew_fit rebuilt by `profiler refit-skew`
  MoE          tp1/moe.csv (all EP slices) as at every TP in the fork
usage: build_tp4proj.py FORK
"""
import copy, json, os, shutil, sys
import pandas as pd, yaml

fork = os.path.abspath(sys.argv[1])
base = os.path.join(fork, "profiler/perf/RTXPRO6000/Qwen/Qwen3-30B-A3B-Instruct-2507")
src, dst = os.path.join(base, "bf16"), os.path.join(base, "bf16-tp4proj")
assert not os.path.exists(dst), dst
shutil.copytree(src, dst)
os.makedirs(f"{dst}/tp4")
for name, ncols in (("dense", None), ("per_sequence", None), ("attention", None)):
    a = pd.read_csv(f"{src}/tp1/{name}.csv"); b = pd.read_csv(f"{src}/tp2/{name}.csv")
    keys = [c for c in b.columns if c != "time_us"]
    m = b.merge(a, on=keys, how="left", suffixes=("", "_tp1"))
    assert len(m) == len(b) and m["time_us_tp1"].notna().all(), name
    ratio = (m["time_us_tp1"] / m["time_us"]).clip(1.0, 2.0)
    out = b.copy(); out["time_us"] = (m["time_us"] / ratio).round(5)
    out.to_csv(f"{dst}/tp4/{name}.csv", index=False)
    print(name, len(out), "mean ratio tp4/tp2", float((out.time_us / b.time_us).mean()))
shutil.copy(f"{dst}/tp2/skew.csv", f"{dst}/tp4/skew.csv")

meta = yaml.load(open(f"{dst}/meta.yaml"), Loader=yaml.CSafeLoader)
meta["variant"] = "bf16-tp4proj"
four = copy.deepcopy(meta["per_tp"][2] if 2 in meta["per_tp"] else meta["per_tp"]["2"])
def retp(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("tp", "tensor_parallel_size") and v == 2:
                o[k] = 4
            else:
                retp(v)
    elif isinstance(o, list):
        for v in o:
            retp(v)
retp(four)
meta["per_tp"]["4"] = four
for e in meta["skew_fit"]["per_tp"].values():
    e["identity"]["variant"] = "bf16-tp4proj"
meta["projection"] = {
    "status": "tp4 is a projection, not a measurement; tp1 and tp2 are the measured files of bf16, copied unchanged",
    "source_variant": "bf16 (this fork)",
    "tp4_rule": "tp2 time / clamp(tp1 time / tp2 time, 1, 2), per row of dense, per_sequence and attention",
    "tp4_skew": "tp2 skew.csv and per_tp skew metadata copied; skew_fit rebuilt with profiler refit-skew",
    "tp4_moe": "none written: MoE lookups use tp1 at every TP",
    "note": "same recipe as LLMServingSim-upstream-a4053bc bf16-tp4proj, re-derived on this fork's measurements "
            "(the a4053bc bundle's format is not loadable here), so its numbers differ from that bundle's",
}
yaml.dump(meta, open(f"{dst}/meta.yaml", "w"), Dumper=yaml.CSafeDumper, sort_keys=False)
