# Migration from the HiPC fork to upstream a4053bc

The old version is the Marvell/HiPC fork: branch `feature/tiered-memory-topology`
at a9b2052, built on upstream 89f312f. The new version is branch
`migration/upstream-a4053bc`, which puts that work on upstream a4053bc. On
2026-09-15 a4053bc was still upstream's newest commit, and the submodules sit on
upstream's newest `astra-sim` (d346994) and `chakra` (30221ab).

Upstream changed a lot in between. `main.py` became `python -m serving`,
`inference_serving/` became `serving/core/`, `cluster_config/` became
`configs/cluster/` and `dataset/` became `workloads/`. The byte-counting memory
model was replaced by a block manager, the scheduler now follows vLLM V1, and
profiles use a new per-category format.

Everything the fork's experiments use runs on the new version, including tiered
P/D KV handoff and the profile projection generator, which the first pass of the
port missed. The transport modes and memory fixes give exactly the same results
as upstream's own path. Results do not match the old fork, because upstream's
scheduler and memory model changed. The migration branches are published but
not merged into `main`; explicit DP groups only work in legacy mode.

## What was ported

| Fork feature | Where it lives now |
|---|---|
| In-memory and shared-template ET transport | `--execution-template-mode`, `serving/core/execution_templates.py` |
| One-line READY/COMPLETE controller protocol | `--compact-controller-protocol` |
| Bounded template cache in ASTRA-Sim | `--template-cache-max-entries` |
| Transport metrics | `--execution-template-stats-output` |
| Tier model: capacity reservation, P/D KV handoff over the fabric, non-local KV read cost | `memory_tiers`, `fabric` and `pd_transfer` in a cluster config, `kv_tier` / `baseline_kv_tier` / `compute_endpoint` per instance, `--tier-stats-output` |
| H100 Llama-3.1-70B tp4/tp8 profiles | `profiler/perf/H100/meta-llama/Llama-3.1-70B/bf16/` |
| Llama-3.1-405B tp8 nominal/low/high projections | `profiler/perf/H100/meta-llama/Llama-3.1-405B/{bf16,bf16-low,bf16-high}/` |
| TP8 and 405B projection generator | `profiler/v0/profile_projection.py`, `profiler/v0/PROFILE_PROJECTIONS.md` |
| Converter for the old profile format | `python -m profiler export-v0`, `profiler/v0/export.py` |
| Profile variant selection | `--profile-variant` |
| Finite-grid attention fallback | `serving/core/trace_generator.py`, used only for converted bundles |
| 405B model and TP8×PP2 deployment | `configs/model/meta-llama/Llama-3.1-405B.json`, `configs/cluster/llama_405b_h100_tp8_pp2.json` |
| Scale runner, resource monitor, run manifests, run comparator, config generator | `analysis/` |
| ASTRA-Sim payload and template feeder | commits on the `astra-sim` submodule |
| Chakra in-memory feeder and template converter | commits on the nested `chakra` submodule |

Some fork work was already in upstream in another form, so it was not ported:

- Logging. Upstream's `serving/core/logger.py` descends from the fork's, and it
  also fixes the ANSI codes in log files that the fork never fixed.
- Compact graph metadata. Upstream stores the trace path in each `.et` instead
  of the trace text, which removes the same overhead as the fork's
  `compact_metadata` flag.
- Converted-graph reuse. Upstream caches whole conversions across batches.
  Shared-template mode skips that cache and deduplicates across ranks instead.
- Pipeline stage boundaries. Upstream cuts stages from the trace header, so the
  fork's `pipeline_boundaries` argument was dropped.

Some was dropped on purpose:

- The fork's managed-controller readiness guard. Upstream fixed the same DP
  hangs another way, and with both in place the controller waited twice: clocks
  rose 17% on `pp`, 43% on `moe_pp`, 57% on `moe_dp_tp_pp` and 87% on
  `moe_dp_tp_pp_uneven`. Without it every baseline matches.
- `--cleanup-consumed-traces` and `--retain-traces`. Upstream deletes per-run
  inputs by default, and `--keep-inputs` keeps them.
- `--template-bundle-builder`. Shared-template mode always uses the fused
  builder. On 2026-09-13 it produced the same bundles as the old payload builder
  for all 718 batches of two runs.

## Fixed during the migration

The fork had these problems too.

ASTRA-Sim leaked per-collective state. A 72-NPU TP8 ShareGPT-750 run grew
ASTRA-Sim to 47.5 GiB in a straight line while Python stayed at 0.2 GiB. The
code is upstream's, so legacy mode leaked the same way. The sites, measured on a
100-request TP8 run with 9.90 GB of heap in use:

| Site | Bug | Share |
|---|---|---|
| `CommonNetworkApi::process_chunk_arrival` | never freed the tuple `sim_send` allocates per chunk | ~80% |
| `ChunkIdGenerator::chunk_id_map` | one entry per collective per rank, never erased | ~10% |
| `Workload::collective_comm_node_id_map` | never erased | ~5% |
| `IntData` from `DataSet::notify_stream_finished` | never deleted | ~4% |
| `BaseStream::synchronizer` / `ready_counter`, `SchedulerUnit` usage history | written, never read | <1% |

After the fixes:

| ShareGPT-750 run | ASTRA-Sim peak RSS | Wall time | `requests.csv` |
|---|---|---|---|
| 70B TP8, 72 NPUs | 47.5 GiB → 0.12 GiB | 58:35 → 42:38 | byte-identical |
| 405B TP8×PP2, nominal | 7.1 GiB → 0.07 GiB | 8:03 → 7:08 | byte-identical |

The fixed runs shared the host with other simulations, so their wall times are
an upper bound. LeakSanitizer on the `tp_pp` scenario went from 5.03 MB leaked at
exit to 0.3 MB. What is left is `Sys`-owned objects that are never torn down,
and it does not grow with run length.

Also fixed:

- Under the compact protocol, ASTRA-Sim console logs go to stderr. A
  RingTopology log line had landed inside a READY record and left both processes
  waiting on each other.
- The frontend raises an error on a malformed READY record or an early EOF
  instead of hanging.
- `decode_base64` shifted a signed int into the sign bit, which UBSan reports as
  undefined. The accumulator is unsigned now.
- The file-free modes refuse `--network-backend ns3` at startup. The fork had
  this guard and the first pass of the port lost it.

Still unbounded, but not a problem at current sizes: `DataSet::id_auto_increment`
is an `int` shared by every rank. The 100-request TP8 run used 11.8M ids, so a
run with about 180 times as many collective completions would overflow it.

## Is it exact?

Against upstream's own behaviour, yes. With the new flags off and no
`memory_tiers` block, `./serving/validate.sh --clocks-only` matches all 58
recorded clock baselines, and the regenerated `bench/examples` `sim.csv` and
`summary.txt` files are byte-identical to upstream's.

Shared-template transport against legacy transport, yes. With
`--execution-template-mode shared-template --compact-controller-protocol
--template-cache-max-entries 128`, all 46 scenarios without a `dp_group` match
the same baselines. On 2026-09-14, three real ShareGPT requests also gave
identical legacy and shared-template CSVs for 405B TP8×PP2 and for 70B TP8 on 72
NPUs.

Converted profiles against their v0 source, yes on every profiled point. The
exporter does no fitting. Dense and per-sequence timings are sums of the v0
operators that vLLM now fuses, and `layernorm` is the mean of v0's two
layernorms, because the new trace emits it twice per block. Units change from
nanoseconds to microseconds. Attention lookups reproduce v0's 32/64-token key
rounding, its L2-norm prefill key and its additive prefill-plus-decode
composition. `tests/test_v0_export.py` checks more than 10,000 fused sums and
checks attention against the v0 tables within 1 ns of rounding.
`tests/test_v0_attention_compat.py` checks the projected bundles' lookups
exactly. Between grid points the lookup uses the fork's bilinear rule, and
outside the grid it extrapolates linearly from the edge.

The v0 data cannot supply three things, and the fork had the same gaps:

- `sampler` is 0, clamped to 1 ns, because v0 never measured it.
- There is no skew correction, because v0 only swept uniform decode batches.
  Decode batches with mixed KV lengths come out optimistic.
- v0's KV axis stops at 2048 tokens. Past that, attention time is extrapolated.

The labels are unchanged. H100 tp4 is derived from measured data. tp8 is
projected from measured tp1/tp2/tp4 and back-tested at 3.89% median and 21.78%
P90 error. 405B is a calibrated projection, never a measurement.

Regenerated projections against the committed ones, yes for 405B.
`project-405b` reproduces the nominal, low and high v0 tables and the memory
report byte for byte, and `export-v0` turns them into the committed runtime
bundles byte for byte. `project-tp8` matches the committed 70B TP8 layers and
decode attention. Its prefill attention differs on 133,056 rows, the same rows
as the fork's own generated output, because the committed TP8 is the fork's
frozen v0 profile. The back-test comes out at the documented 3.89% and 21.78%.
`tests/test_profile_projection.py` checks this on every run.

Tiered P/D handoff against the fork: same mechanism, different makespan. On the
fork's 2-request switched-CXL mechanism test, the new version makes the same 379
remote KV reads, adds 95.08 ms of extra read latency (fork: 95.68 ms) and moves
6.20 GB over the CXL link (fork: 6.24 GB). Makespan is 7.73 s against the fork's
20.13 s, because upstream's scheduler and timing changed. CXL still costs more
than local HBM (7.73 s against 7.63 s). Legacy and shared-template runs give
identical CSVs and tier stats at 2 and 20 requests, every reservation is
released by the end, and none fails. `tests/test_pd_tiered_scheduler.py` covers
the handoff, the decode wait and the release.

Against the old fork's results, no. The profiles match, but upstream's scheduler
batches differently. `--max-num-seqs` defaults to 128 where the fork had no cap.
Prefix caching, chunked prefill and full-ISL reservation are on by default. KV
capacity comes from the block manager: 807,552 tokens per rank at 0.90
utilization for 405B, against 628,358 in the fork's feasibility accounting.

405B nominal on ShareGPT-750:

| Setup | req/s | Mean / p99 TTFT (ms) | Mean / p99 TPOT (ms) |
|---|---|---|---|
| Fork | 5.31 | 389 / 1,119 | 148.9 / 263.8 |
| New, upstream defaults | 4.68 | 17,517 / 41,798 | 76.6 / 91.7 |
| New, `--max-num-seqs 1024 --no-enable-prefix-caching` | 6.01 | 196 / 528 | 98.3 / 138.9 |
| Same, plus `--no-enable-chunked-prefill` | 6.02 | 190 / 507 | 97.6 / 134.2 |

The 17.5 s TTFT comes from the 128-sequence cap. Lifting it brings TTFT under
the fork's, and chunked prefill makes no difference. TPOT stays about a third
below the fork's even with the settings matched. The likely cause is pipeline
scheduling. The fork took a request out of its queue until its batch finished,
so at PP=2 a request advanced once per full pipeline pass. Upstream follows vLLM
and lets a request sit in both in-flight microbatches. That comes from reading
both schedulers; no run has isolated it.

The full 405B uncertainty band. Each cell is new version with upstream defaults /
new version with the fork's settings / fork:

| Arm | req/s | Mean TTFT (ms) | Mean TPOT (ms) |
|---|---|---|---|
| low | 5.84 / 6.81 / 6.37 | 7,785 / 136 / 231 | 60.4 / 68.4 / 94.9 |
| nominal | 4.68 / 6.01 / 5.31 | 17,517 / 196 / 389 | 76.6 / 98.3 / 148.9 |
| high | 3.90 / 5.23 / 4.54 | 27,361 / 269 / 649 | 92.5 / 131.6 / 209.2 |

The arms keep their order in every setup. With the fork's settings the band sits
lower and is narrower than the fork's: high/low TPOT is 1.9x against 2.2x.

## Differences and gaps

**Tiered P/D works differently from the fork in three small ways.** A cluster
config with `memory_tiers` needs `--no-enable-prefix-caching`, or the run stops
at startup. The fork refused the combination too, but upstream turns prefix
caching on by default, so this is now a flag you have to pass. Under the tier
model the prefill instance skips upstream's per-layer network KV send, because
the fabric handoff already charges the transfer. Without tiers that send is
unchanged. The handed-off KV is rounded to blocks the upstream way, which comes
out one block smaller than the fork when a prompt is an exact multiple of the
block size.

**Fork documentation.** `HANDOFF.md`, `HIPC_TODOS.md`, `llmservingsim_context.md`,
`TIERED_MEMORY.md` and `TP8_PROFILE.md` stay in the fork. Why: they describe the
fork's code paths and command lines, which no longer run as written. They are
still the place to read for background. `PROFILE_PROJECTIONS.md` came over with
the generator and is rewritten for the new paths.

**Explicit DP groups in the file-free modes.** `--execution-template-mode
in-memory` and `shared-template` reject any cluster config with a `dp_group` at
startup. Why: the fork never had DP groups, so there was nothing to port.
Upstream's DP barrier writes every member's graph into one shared workload
folder. The file-free path sends each member's graph separately, and the round
never completed. A startup error replaced the hang. Use legacy mode for DP. None
of the fork's configs use `dp_group`.

**Removed flags with no replacement.** `--prioritize-prefill` and
`--enable-attn-prediction`. Upstream a4053bc no longer has either feature.

**Two cluster configs.** `baseline_scale_96.json` and
`single_node_single_instance_H100_tp8.json` were never committed in the fork and
use the old config fields. Regenerate them with
`analysis/generate_scale_configs.py --npus 96`, or `--npus 8 --tp 8`.

**Published on migration branches.** `.gitmodules` points `astra-sim` at
`github.com/aryanbethman/astra-sim`, and `astra-sim` points `chakra` at
`github.com/aryanbethman/chakra`. All three `migration/upstream-a4053bc`
branches are published. A fresh recursive clone was verified to fetch the exact
LLMServingSim, ASTRA-Sim and Chakra commits listed below. The migration has not
been merged into `main`.

## Running it

Use `./env/bin/python` and put `env/bin` first on `PATH`. The machine-wide
editable `chakra` install points at the fork's checkout and would shadow this
tree's copy.

ASTRA-Sim links protobuf 3.6.1 from `env/cpp`, a conda prefix inside this
tree (ignored by git, like the rest of `env/`). Before 2026-09-19 the build
had picked up the fork's `~/LLMServingSim/env` instead, so the binary carried
a RUNPATH into the other checkout. The prefix holds the same conda-forge
builds the fork used, byte for byte, so creating it needs no download when
conda's package cache has them:

    ~/miniconda3/bin/conda create -y --offline -p "$PWD/env/cpp" -c conda-forge -c defaults \
      libprotobuf=3.6.1=hdbcaa40_1001 libstdcxx-ng=11.2.0=he4da1e4_16 \
      zlib=1.2.13=h4ab18f5_6 libzlib=1.2.13=h4ab18f5_6 \
      libgcc=15.1.0=h767d61c_5 libgcc-ng=15.1.0=h69a702a_5 \
      libgomp=15.1.0=h767d61c_5 _openmp_mutex=5.1=1_gnu _libgcc_mutex=0.1=main

Drop `--offline` on a machine without those packages cached. Then configure
against it from a clean cache:

    cd astra-sim/build/astra_analytical/build
    rm -rf CMakeCache.txt CMakeFiles
    PATH="$OLDPWD/env/cpp/bin:$PATH" cmake .. -DBUILDTARGET=all -DCMAKE_PREFIX_PATH="$OLDPWD/env/cpp"
    cmake --build . -j 16

`ldd build/bin/AstraSim_Analytical_Congestion_Unaware` should resolve
`libprotobuf.so.17` and `libstdc++.so.6` under `env/cpp/lib`.

A quick 405B run:

    ./env/bin/python -m serving \
      --cluster-config configs/cluster/llama_405b_h100_tp8_pp2.json \
      --profile-variant bf16 --dtype bfloat16 --block-size 16 \
      --dataset workloads/example_trace.jsonl --num-reqs 10 \
      --execution-template-mode shared-template \
      --compact-controller-protocol --template-cache-max-entries 32

A tiered P/D run over switched CXL:

    ./env/bin/python -m serving \
      --cluster-config configs/cluster/tiered_memory_pd_cxl_h100_tp4.json \
      --dataset experiments/hipc_upstream_control/sharegpt_req750_rate10_llama.jsonl \
      --num-reqs 20 --no-enable-prefix-caching --tier-stats-output tier.json

A monitored ShareGPT-750 run that writes a manifest. `PROFILE_VARIANT` picks the
uncertainty arm and is recorded in the manifest; leave it unset for the
dtype-derived bundle:

    PROFILE_VARIANT=bf16-low analysis/run_scale_experiment.sh RESULT_DIR \
      configs/cluster/llama_405b_h100_tp8_pp2.json 16 128

To regenerate the TP8 and 405B profiles, see `profiler/v0/PROFILE_PROJECTIONS.md`.

An intermittent hang with both processes blocked on the pipe was seen on
2026-09-13 while several runs shared one checkout, and it was never root-caused.
Several concurrent runs on 2026-09-15 finished without it. Use a timeout on long
runs.

Old command lines need these changes:

| Fork | New |
|---|---|
| `python main.py` | `python -m serving` |
| `--fp 16` | `--dtype bfloat16` |
| `--num-req N` | `--num-reqs N` (the old spelling still parses as a prefix) |
| `--max-batch N` (default 0, no cap) | `--max-num-seqs N` (default 128) |
| `--gen` | `--skip-prefill` |
| `--profile-variant nominal\|low\|high` | `--profile-variant bf16\|bf16-low\|bf16-high` |
| `--template-bundle-builder`, `--cleanup-consumed-traces`, `--retain-traces` | removed, see above |
| `--prioritize-prefill`, `--enable-attn-prediction` | removed, no replacement |
| `cluster_config/`, `dataset/` | `configs/cluster/`, `workloads/` |

Cluster configs use `num_npus`, `tp_size` and `pp_size` instead of the fork's
`npu_num`, `npu_group` and `pipeline_parallel_degree`, so fork configs need
their instance fields rewritten. Some defaults changed as well: prefix caching
is on (`--no-enable-prefix-caching` turns it off), request routing is `LOAD`
instead of `RR`, expert routing is `BALANCED` instead of `FAST`, and
`--log-interval` is 1.0 s instead of 0.5 s.

## Validating a change

    ./serving/validate.sh --clocks-only

This runs 58 scenarios against recorded clocks and requires exact equality.
Flags that only change how graphs reach ASTRA-Sim must not move a clock, so
check them against the same baselines. Pass the 46 scenario names whose configs
have no `dp_group`; the other 12 fail at startup on purpose:

    EXTRA_ARGS="--execution-template-mode shared-template --compact-controller-protocol --template-cache-max-entries 128" \
      PYTHON="$PWD/env/bin/python" ./serving/validate.sh --clocks-only single tp_pp multi_instance_8npu

No scenario has a `memory_tiers` block, so run the tiered P/D command above in
both modes after touching the tier path and compare the CSVs.

Unit tests, no GPU or vLLM needed:

    ./env/bin/python -m unittest discover -s tests

Without `--clocks-only`, `validate.sh` also regenerates `bench/examples`. `env/`
has no matplotlib, so that stage reports `Accuracy: FAIL` on the import alone.
Run it with a Python that has matplotlib: `PYTHON=/path/to/python
./bench/examples/validate.sh`.

State on 2026-09-15, with the tiered P/D port: 51 unit tests pass, and 58/58
legacy and 46/46 shared-template clock scenarios match. The bench outputs were
checked byte-identical before that port, which changes nothing without
`memory_tiers`.

## Where the code is

| Repo | Branch | Built on |
|---|---|---|
| LLMServingSim | `migration/upstream-a4053bc` | upstream a4053bc |
| `astra-sim` | `migration/upstream-a4053bc` | upstream d346994, plus the fork's feeder commits, the chakra bump, compact-protocol logging, the leak fixes, the base64 fix and the chakra fork URL |
| `chakra` | `migration/upstream-a4053bc` | upstream 30221ab, plus the fork's feeder and converter commits |

All three branches are published. A recursive clone resolves the pinned
submodule commits without using the original anjuna3 checkout. The migration is
still isolated from `main` pending review and an explicit merge decision.
