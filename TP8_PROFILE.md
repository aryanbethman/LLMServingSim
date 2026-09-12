# How the TP=8 profile was made

The H100/Llama-3.1-70B TP=8 profile is **projected from the measured tp1, tp2, and tp4
sweeps**. No H100 was available, so nothing in it was measured at TP=8. Traces generated
from it are structurally real — 8-rank Chakra ET bundles that ASTRA-sim consumes normally —
but the latencies inside them are estimates.

Use it for simulator-scaling work. Do not use it for a serving-latency claim.

## Why not just divide tp1 by 8

That was the obvious idea and it doesn't work. Dividing the measured tp1 by 4 and comparing
against the real measured tp4 comes out 17–27% too fast, depending on sequence length, and
70% of individual rows are underestimates.

The reason is that only some layers shard. At long sequences the projections
(`q_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head`) hit 4.1–4.4x, so
dividing by 4 is right for them. The layernorms and `embedding` sit at 1.06x — they do
identical work regardless of GPU count, and dividing them invents a speedup that doesn't
exist. `k_proj` and `v_proj` only reach 2.15x and 2.34x, because Llama-3.1-70B has 8 KV
heads total; at TP=8 each rank holds exactly one and the GEMM is too small to fill the GPU.

## The method

For each `(layer, sequence length)` row, take the speedup the last measured doubling
actually delivered (`tp2 -> tp4`), clamp it to `[1.0, 2.0]`, and apply it once more to the
tp4 latency. The clamp matters both ways: sharding across 2x the devices cannot beat a 2x
speedup, and cannot make a layer slower.

That raw version is optimistic, because TP scaling decelerates — `attn` went 1.76x then
1.53x across the two measured doublings, so "the next doubling repeats the last one"
overshoots. The bias is measurable on the one doubling that can be checked: predict tp4
from tp2 using the `tp1 -> tp2` ratio, compare against real tp4, and divide the per-layer
result back out. Worst offenders were `rope` (-12.4%), `attn` (-11.0%), `v_proj` (-7.9%),
and `k_proj` (-7.4%). Correcting moved median signed error from -1.3% to 0.0%.

Three candidate models were tested against ground truth rather than assumed. An
overhead-split model (`t8 = floor + (t4 - floor) / 2`) looked physically sound but came out
64% off at short sequences, so it was dropped for the layers.

**Prefill attention** is scaled from tp4 by a factor that varies with `prefill_chunk_size`,
read off the measured `attn` rows. The factor is genuinely chunk-dependent — 1.00x at small
chunks where the kernel is launch-bound, up to 1.78x at 2048 — so a single multiplier would
have been wrong. Beyond 2048 it is held at its top-of-range value.

**Decode attention is the weak part.** No per-TP decode measurements exist at any TP, so
this is a physical model, not a fit: head-parallel sharding halves the KV bytes each rank
reads (2 heads at TP=4 down to 1 at TP=8) on top of a fixed 2594 ns overhead floor, then
de-biased by the `attn` figure. It is unvalidated. If a claim rests on decode attention
latency at TP=8, this profile does not support it.

## How accurate it is

Backtested by predicting tp4 from tp2 and comparing against ground truth:

| | median error | p90 abs error | whole-model, short seq | whole-model, long seq |
|---|---|---|---|---|
| naive divide-by-2 | — | — | -18.8% | -11.4% |
| ratio method | -1.3% | 21.8% | -0.5% | 0.0% |
| ratio + bias correction | 0.0% | 19.8% | — | — |

Per-layer absolute error is in `tp8_v1_generated/calibration/tp4_backtest_by_layer.csv`.
It ranges from 1.1% median for `lm_head` up to 20.9% for `attn`, which is the layer the
method handles worst and also the one that matters most.

The resulting TP=4 to TP=8 speedup is **1.47x–1.56x** depending on sequence length, not 2x.

Every backtest error before correction ran in the same direction, optimistic. Treat these
numbers as a **floor on latency**; real hardware would likely be slower. TP=8 also sits
outside the measured range, and the bias correction assumes the per-doubling bias is
constant, which cannot be checked with only two measured doublings.

## What is in the tree

- `tp8/` — the active profile. Byte-identical to `tp8_v0_reference/`.
- `tp8_v0_reference/` — preserved copy of the original generation.
- `tp8_v1_generated/` — a later reproducible regeneration carrying
  `profile_manifest.json` (source hashes, method parameters,
  `measurement_status: projected_not_measured`), `validation.json`, and the backtest
  artifacts.

v1 reproduces v0 exactly for `layers.csv` and decode attention. It differs only in prefill
attention rows where `prefill_chunk_size > 2048` — 133,056 rows, all of them beyond the
measured range — where it reads about 2.3% lower. Everything at or below 2048 is identical.

## Replacing it with measured data

One H100 for an afternoon is enough. The profiler never loads the real model: it builds
`LlamaForCausalLM(config)` with random weights and `num_hidden_layers = 1`, divided by
`tp_size`, which is 0.74 GB at TP=8 rather than 140 GB. Communication is not measured here
at all — the all-reduces are priced separately by ASTRA-sim's network model — so the eight
ranks never need to talk and one GPU suffices.

Run a **tp4 control first**, writing to `--hardware H100-verify` so it cannot clobber the
reference. If per-layer medians don't match the existing tp4, the rented card isn't
comparable to whatever measured tp1–tp4, and mixing a measured tp8 into that set would be
worse than this projection. Then run the three profiler commands from `llm_profile/` at
`--tp-size "8"`, matching the tp4 flags (`--max-len 2048`, `--warmup 10`, layers
`--repeat 30`, attention `--repeat 50`).

If the afternoon runs short, use `--profile-only-decode`. Decode attention is the only
piece with no ground truth behind it.

Afterwards, delete `tp8/predictions/*.pkl`. `_load_attn_perf_db_dict` prefers the pickles
over the CSVs, so stale ones would silently shadow real measurements.
