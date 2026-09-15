# tp8 — EXTRAPOLATED, NOT MEASURED

These files were **not** produced by running the profiler on an H100. They were
synthesised from the measured `tp1/`, `tp2/`, and `tp4/` sweeps because no H100 was
available. Do not report results built on this profile as measured TP=8 hardware data.

Generated 2026-09-06. Replace this directory with a real profiler run if H100 time
becomes available; nothing downstream needs to change.

## Method

**`layers.csv`** — for each `(layer, sequence length)` row, take the speedup that the
last measured doubling actually delivered (`tp2 -> tp4`, clamped to `[1.0, 2.0]`) and
apply it once more to the tp4 latency. The clamp matters in both directions: sharding
across 2x more devices cannot exceed a 2x speedup, and it cannot make a layer slower.

The raw method is optimistic, because TP scaling decelerates. That bias was measured on
the one doubling that can be checked — predict `tp4` from `tp2` using the `tp1 -> tp2`
ratio, compare against real `tp4` — and divided back out per layer.

| Layer | Measured bias | | Layer | Measured bias |
|---|---|---|---|---|
| `attn` | −11.0% | | `o_proj` | −1.8% |
| `rope` | −12.4% | | `final_layernorm` | −1.5% |
| `v_proj` | −7.9% | | `input_layernorm` | −1.3% |
| `k_proj` | −7.4% | | `post_layernorm` | −1.3% |
| `embedding` | −2.1% | | `down_proj` | −0.9% |
| `act_fn` | +5.6% | | `lm_head` | +0.5% |
| `q_proj` | +1.8% | | `gate_proj` | +1.3% |
| `up_proj` | +1.4% | | | |

**`predictions/attn_prefill_predictions.csv`** — scaled from tp4 by a factor that varies
with `prefill_chunk_size`, read off the measured `attn` rows in `tp2`/`tp4`. The factor is
genuinely chunk-dependent (1.00x at small chunks where the kernel is launch-bound, up to
1.78x at 2048), so a single multiplier would have been wrong. Beyond 2048 the factor is
held at its top-of-range value.

**`predictions/attn_decode_predictions.csv`** — **weakest link.** No per-TP decode
attention measurements exist at any TP, so this is a physical model rather than a fit:
head-parallel sharding halves the KV bytes each rank reads (2 heads at TP=4 to 1 at TP=8)
on top of a fixed overhead floor of 2594 ns, i.e. `t8 = floor + (t4 - floor) / 2`, then
de-biased by the `attn` figure above. **This is unvalidated.** If a claim depends on
decode attention latency at TP=8, this profile does not support it.

## Accuracy

Validated by predicting `tp4` from `tp2` and comparing against ground truth:

| | median error | p90 abs error | whole-model, short seq | whole-model, long seq |
|---|---|---|---|---|
| naive divide-by-2 | — | — | −18.8% | −11.4% |
| ratio method | −1.3% | 21.8% | −0.5% | 0.0% |
| ratio + bias correction | 0.0% | 19.8% | — | — |

Resulting TP=4 to TP=8 speedup is **1.47x–1.56x** depending on sequence length, not 2x.
Layernorms and embedding stay flat (~1.02x) because they do not shard; `k_proj`/`v_proj`
reach only ~1.37x because Llama-3.1-70B has 8 KV heads total, so at TP=8 each rank holds
exactly one and the GEMM is too small to fill the GPU.

## Known limitations

- Every validation error was one-directional (optimistic) before correction. Treat these
  numbers as a **floor on latency**; real hardware would likely be somewhat slower.
- TP=8 is outside the measured range, and the trend is decelerating. The bias correction
  assumes the per-doubling bias is constant, which cannot be checked with only two
  measured doublings.
- Decode attention is modelled, not measured or fitted (see above).
- Source sweeps cover sequence lengths 1–2048 with `kv_cache = 0`, 30,720 rows per TP.

## Reproducing

The generator lives outside the repo (it was a one-off) and reads only `tp1/`, `tp2/`, and
`tp4/`. The whole method is the two paragraphs above; it is about 30 lines of pandas.
