# TP8 and 405B H100 profile projections

These profiles are projections, not measurements. The inputs are the measured
H100 Llama-3.1-70B TP=1, TP=2 and TP=4 profiles under
`perf_models/H100/meta-llama/Llama-3.1-70B/`. `profile_projection.py` writes the
TP=8 and 405B profiles from them in the old v0 layout, and
`python -m profiler export-v0` converts those into the bundles the simulator reads
under `profiler/perf/`. The generator never loads model weights or uses a GPU.

## 70B TP=8

`project-tp8` scales each TP=4 row by the TP=2 to TP=4 speedup, clamped to 1-2x,
and corrects each layer by the bias measured when predicting TP=4 from TP=1 and
TP=2. That held-out back-test has 3.89% median and 21.78% P90 absolute error,
inside the 10%/25% gates. It checks the method. It does not make TP=8 a
measurement.

The committed `tp8/` is the frozen v0 profile the fork used, and the simulator
keeps using it. A fresh `project-tp8` run matches its layer rows and decode
attention byte for byte. Prefill attention differs: v0's long-prefill scaling was
never written into the old profile, so the generator uses the nearest measured
scaling instead. The fork's own generated output has the same 133,056 changed
prefill rows.

    ./env/bin/python profiler/v0/profile_projection.py project-tp8 --output /tmp/tp8_v1
    ./env/bin/python profiler/v0/profile_projection.py compare \
      --reference profiler/v0/perf_models/H100/meta-llama/Llama-3.1-70B/tp8 \
      --generated /tmp/tp8_v1 --output /tmp/tp8_v1/v0_comparison.json

## Llama-3.1-405B TP=8

`project-405b` transforms each measured 70B TP=4 operator by its sharded FLOP and
HBM-byte geometry at 405B TP=8, using `configs/model/meta-llama/Llama-3.1-405B.json`.
Attention scales with per-rank heads. `variants/low` and `variants/high` apply
the 70B back-test's P90 error to layer and attention latency as a sensitivity
band. It also writes a TP=8×PP=2 memory report: 50.73 GB of BF16 weights and
32,256 KV bytes per token per GPU, which fits a 128K-token context with an 8 GB
runtime and 1 GB communication reserve.

Regenerating gives the committed nominal, low and high tables and the memory
report byte for byte, and converting them gives the committed runtime bundles
byte for byte:

    ./env/bin/python profiler/v0/profile_projection.py project-405b --output /tmp/405b
    ./env/bin/python -m profiler export-v0 meta-llama/Llama-3.1-405B --from /tmp/405b \
      --hardware H100 --tp 8 --variant bf16 --out-root /tmp/perf
    ./env/bin/python -m profiler export-v0 meta-llama/Llama-3.1-405B --from /tmp/405b/variants/low \
      --hardware H100 --tp 8 --variant bf16-low --out-root /tmp/perf
    ./env/bin/python -m profiler export-v0 meta-llama/Llama-3.1-405B --from /tmp/405b/variants/high \
      --hardware H100 --tp 8 --variant bf16-high --out-root /tmp/perf

Call any result from these profiles a calibrated projection.

`tests/test_profile_projection.py` regenerates both profiles into a temporary
directory and checks them against the committed files. `profile_projection_report.py`
draws the back-test plot. It needs matplotlib, which `env/` does not have, so run
it with another Python:

    python3 profiler/v0/profile_projection_report.py --profile /tmp/tp8_v1 \
      --output-dir /tmp/tp8_v1/calibration
