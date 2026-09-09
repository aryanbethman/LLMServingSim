# Reproducible TP=8 and 405B H100 projections

This directory contains projections, not newly measured hardware profiles.  The
source measurements are the checked-in H100/Llama-3.1-70B TP=1, TP=2, and TP=4
layer/attention profiles.

## 70B TP=8

`tp8_v0_reference/` is a frozen copy of the previously used TP=8 profile.
`tp8_v1_generated/` is generated deterministically by
`llm_profile/profile_projection.py`.  Layer rows and decode-attention rows are
semantically identical to v0.  Prefill rows through 2,048 tokens are also
identical.  v0's long-prefill (>2,048-token) held scaling factor was not encoded
in the old profile, so v1 records a transparent nearest-measured scaling rule
instead; `v0_comparison.json` records every differing row.  v0 remains the
canonical compatibility profile: v1 is not silently substituted for it.

Reproduce v1 and its comparison:

```bash
env/bin/python3 llm_profile/profile_projection.py project-tp8 \
  --output llm_profile/perf_models/H100/meta-llama/Llama-3.1-70B/tp8_v1_generated \
  --method-version v1
env/bin/python3 llm_profile/profile_projection.py compare \
  --reference llm_profile/perf_models/H100/meta-llama/Llama-3.1-70B/tp8_v0_reference \
  --generated llm_profile/perf_models/H100/meta-llama/Llama-3.1-70B/tp8_v1_generated \
  --output llm_profile/perf_models/H100/meta-llama/Llama-3.1-70B/tp8_v1_generated/v0_comparison.json
```

The held-out TP=4 back-test has 3.89% median absolute error and 21.78% P90
absolute error, within the 10%/25% acceptance gates.  It supports the projection
method; it does **not** turn TP=8 into a physical H100 measurement.

## Llama 3.1 405B TP=8

`model_config/meta-llama/Llama-3.1-405B.json` records the architecture used for
analytical shapes: 126 layers, hidden size 16,384, intermediate size 53,248,
128 attention heads, 8 KV heads, BF16, and 131,072 maximum context.  No model
weights are loaded by the generator.

The 405B profile transforms each measured 70B TP=4 operator using its actual
sharded FLOP and HBM-byte geometry at 405B TP=8.  Attention scales with per-rank
head dimension/heads; ASTRA continues to model collective communication.  The
`variants/low` and `variants/high` profiles apply the 70B held-out P90 residual
as a sensitivity interval to both layer and attention latency.

The memory report assumes **TP=8 × PP=2** on 16 H100-80GB GPUs: 50.73 GB BF16
weights/GPU, 32,256 KV bytes/token/GPU, and 4.23 GB/GPU for one 128K-token
context.  It is feasible with the configured 8 GB runtime plus 1 GB
communication reserve.  A TP=8-only, eight-GPU serving run is intentionally
rejected because it cannot hold the complete model.

The initial PP=2 execution path is implemented in
`cluster_config/llama_405b_h100_tp8_pp2.json`.  It gives Chakra exact
transformer-block boundaries, so the stage-0 ET sends after `down_proj_756` and
stage 1 starts at `input_layernorm_757`; it does not split a transformer block.
A 16-NPU, one-request ShareGPT-750 control completed in 6,092,460,472 simulated
ns, with byte-equivalent request results in legacy and shared-template modes.
The first full 750-request nominal workload completed with exit 0 in 3m47.040s
simulated time (5.31 req/s); its output contains all 750 request records.

The source attention table has a finite batch/KV grid.  When serving produces a
valid point outside that grid, the simulator now retains every exact table row
and uses a cached bilinear interpolation or edge-linear extrapolation only for
the missing point.  The generated request result is therefore still a calibrated
projection and must be labeled accordingly.

Reproduce:

```bash
env/bin/python3 llm_profile/profile_projection.py project-405b \
  --output llm_profile/perf_models/H100/meta-llama/Llama-3.1-405B/tp8 \
  --method-version v1
env/bin/python3 llm_profile/profile_projection_report.py \
  --profile llm_profile/perf_models/H100/meta-llama/Llama-3.1-70B/tp8_v1_generated \
  --output-dir llm_profile/perf_models/H100/meta-llama/Llama-3.1-70B/tp8_v1_generated/calibration
env/bin/python3 -m unittest tests.test_profile_projection -v
```

Every generated directory has a `profile_manifest.json` with input hashes,
generator source/Git hashes, calibration status, parameters, and validation.
Plots and results must call the 405B output a **calibrated projection** until a
physical H100 (or comparable) measurement is available.
