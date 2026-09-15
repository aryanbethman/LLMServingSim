# Fork migration onto upstream a4053bc

The Marvell/HiPC fork (`feature/tiered-memory-topology`) was built on upstream
89f312f. Upstream has moved 129 commits since, restructuring `inference_serving/`
into `serving/`, replacing the byte-counting memory model with a block manager,
and rewriting the profile format. This is the fork's work rebased onto that.

Read `HANDOFF.md` in the fork for what the work is and why. This file is only
about what changed in the move.

## Ported

| What | Where |
|---|---|
| v0 → current profile converter | `profiler/v0/export.py`, `python -m profiler export-v0` |
| H100 / Llama-70B tp4 + tp8 bundles | `profiler/perf/H100/meta-llama/Llama-3.1-70B/bf16/` |
| Llama-405B nominal / low / high | `profiler/perf/H100/meta-llama/Llama-3.1-405B/{bf16,bf16-low,bf16-high}/` |
| 405B model + TP8×PP2 cluster config | `configs/model/meta-llama/Llama-3.1-405B.json`, `configs/cluster/llama_405b_h100_tp8_pp2.json` |
| Uncertainty-arm selection | `--profile-variant` |
| Shared execution templates | `serving/core/execution_templates.py`, `--execution-template-mode` |
| Compact controller protocol | `--compact-controller-protocol` |
| Bounded template reclamation | `--template-cache-max-entries` |
| Topology-aware memory tiers | `serving/core/tiered_memory.py`, `memory_tiers` / `fabric` / `pd_transfer` in a cluster config |
| Aggregate instrumentation | `--tier-stats-output`, `--execution-template-stats-output` |
| Scale config generator | `analysis/generate_scale_configs.py` |
| Resource monitoring, run manifests, run comparison | `analysis/` |
| ASTRA-Sim payload/template feeder, compact-protocol logging, host-memory fixes | commits on the `astra-sim` submodule |
| Chakra in-memory feeder + template converter | commits on the nested `chakra` submodule |

## Already upstream — do not re-port

**Logging.** Upstream's `serving/core/logger.py` is a descendant of the fork's:
same `PROJECT_ROOT_LOGGER_NAME`, same `configure_logger(level, *, log_file)`,
same `ComponentLoggerAdapter`. It adds a Rich console handler and a plain-text
file formatter, which fixes the ANSI-codes-in-logfile problem the fork's own
comment flagged and never fixed.

**Compact graph metadata.** The fork added a `compact_metadata` flag so the
converter stored a trace digest instead of the whole trace text in every rank's
`.et`. Upstream stores the trace *path* instead, which removes the same
overhead (~70% of a 117 KB `.et` on the swe-bench MoE example) unconditionally.

**Converted-graph reuse.** Upstream caches a whole conversion keyed on the trace
rows, so a repeated batch skips the converter. That is a different axis from
shared templates: the cache deduplicates across *time*, templates deduplicate
across *ranks*, and only the latter stops the per-batch cost growing with NPU
count. The converted-graph cache is used in legacy mode; shared-template
mode returns before that cache and uses structural-template deduplication.

**Pipeline-stage boundaries.** Upstream cuts stages from the trace header;
the fork threaded a `pipeline_boundaries` argument through every `convert_*`
method. Upstream's version is kept and the fork's was dropped.

## Dropped

**The fork's managed-controller readiness guard.** It made a controller NPU wait
for all its managed systems before reporting READY. Upstream fixed that class of
bug separately (`Fix DP-group hangs with tp>1 or pp>1`, `Fix three single-slot
assumptions that hung dp>1 with pp>1`, `Skip already-finished requests in
add_done`), and applying both makes the controller wait twice. Measured with it
in place: `pp` +17.1%, `moe_pp` +42.7%, `moe_dp_tp_pp` +57.3%,
`moe_dp_tp_pp_uneven` +87.0%. Removing it restores every PP baseline exactly.

## What the converted profiles cannot tell you

The exporter is deterministic. Dense timings are sums, means, or unit
conversions of the v0 data. Converted bundles additionally carry complete
`attention_prefill_v0.csv` / `attention_decode_v0.csv` surfaces and opt into
`v0_export.attention_lookup: v0-additive`. Runtime uses the original 32/64
key quantization and the prefill L2 query-length key, then adds independent
prefill/decode lookups. The retained coarse 4D table is not the runtime timing
source for these bundles. Native profiles remain on upstream's 4D path.
Tests cover long prefill, mixed batches, uncertainty variants and between-grid
lookups (`tests/test_v0_attention_compat.py`). Two things it cannot recover, recorded in
each bundle's `meta.yaml` rather than guessed:

- **`sampler` is 0.0 (runtime minimum 1 ns).** The v0 profiler never measured
  it. No universal H100 correction is supported by that absence. The exporter
  accepts `--sampler-us` for an explicitly asserted modeling assumption;
  this is not a serving CLI option or a hardware measurement.
- **`skew_fit` is disabled.** v0 only swept uniform decode batches, so there is
  no alpha to fit and the simulator falls back to no skew correction. Converted
  bundles model uniform-batch attention only and are optimistic on
  heterogeneous-KV batches.

The source KV axis stops at 2048, but its prefill-chunk axis extends to
131072. Outside the complete source surfaces, timing is linear extrapolation
from the nearest edge samples, not measurement. Synthesized fallback points
are not inserted into the immutable source tables.

The labels stand as they did in the fork. H100 tp4 is derived from measured
data. tp8 is projected from measured tp1/tp2/tp4, back-tested at 3.89% median
and 21.78% P90. 405B is a calibrated projection, never a measurement.

## ASTRA-Sim host memory at scale

The 72-NPU TP8 ShareGPT-750 run grew ASTRA-Sim to 47.5 GiB, in a straight
line (about 13.4 MB/s) for the whole hour, while Python stayed at 0.2 GiB and
the template cache stayed at 128 entries with no blocked evictions. The growth
was per-collective state that ASTRA-Sim allocated and never released. All of
it is upstream ASTRA-Sim code, so legacy mode leaked the same way.

Two tools found it. An instrumented build printed container sizes at exit
(no sudo or ptrace here, so no heaptrack or live gdb), and a LeakSanitizer
build ran `tp_pp`. For a 100-request TP8 run with 9.90 GB of heap in use:

| Site | Bug | Share |
|---|---|---|
| `CommonNetworkApi::process_chunk_arrival` | never freed the tuple `sim_send` allocates for every chunk | ~80% |
| `ChunkIdGenerator::chunk_id_map` | one entry per collective per rank, never erased; tags are stream ids, so keys are rarely seen again | ~10% |
| `Workload::collective_comm_node_id_map` | never erased | ~5% |
| `IntData` from `DataSet::notify_stream_finished` | never deleted by `Workload::call` | ~4% |
| `BaseStream::synchronizer` / `ready_counter`, `SchedulerUnit` usage history | write-only; nothing reads them but a never-called function | <1% |

The fixes take ownership of the chunk tuple, erase a chunk-id key once no
chunk under it is tracked (both ids then restart at 0 with nothing in flight
to collide with), erase the node-id entry and delete the `IntData` when a
collective completes, drop the stream-id counters with the stream, and delete
the usage history. None of it touches timing: every clock baseline still
matches. LeakSanitizer on `tp_pp` went from 5.03 MB in 164,336 allocations
leaked at exit to about 0.3 MB of `Sys`-owned objects that are never torn down,
which does not grow with run length.

Full ShareGPT-750 reruns with the fixed binary produced byte-identical
`requests.csv` files and identical summary metrics:

| Run | ASTRA-Sim peak RSS | Wall time |
|---|---|---|
| 70B TP8, 72 NPUs | 47.5 GiB → 0.12 GiB | 58:35 → 42:38 |
| 405B TP8×PP2, nominal | 7.1 GiB → 0.07 GiB | 8:03 → 7:08 |

The fixed runs shared the host with other simulations, so their wall times are
an upper bound.

UBSan also flagged the fork's `decode_base64`: a left shift of a negative
`int`. The accumulator is unsigned now.

Still unbounded, but not a problem at these sizes: `DataSet::id_auto_increment`
is an `int` shared by every rank. The 100-request TP8 run used 11.8M ids, so a
run with about 180 times that many collective completions would overflow it.

## Running it

    ./env/bin/python -m serving \
      --cluster-config configs/cluster/llama_405b_h100_tp8_pp2.json \
      --profile-variant bf16 --dtype bfloat16 --block-size 16 \
      --dataset workloads/example_trace.jsonl --num-reqs 10 \
      --execution-template-mode shared-template \
      --compact-controller-protocol --template-cache-max-entries 32

The venv at `env/` exists because the machine-wide editable `chakra` install
points at the fork's checkout and shadows this tree's pinned copy. It carries
the pinned chakra as a package link; see `env/lib/python3.13/site-packages/chakra`.

For a monitored ShareGPT-750 run with a manifest, use the scale runner.
`PROFILE_VARIANT` picks an uncertainty arm and is recorded in the manifest;
leave it unset for the dtype-derived bundle:

    PROFILE_VARIANT=bf16-low analysis/run_scale_experiment.sh RESULT_DIR \
      configs/cluster/llama_405b_h100_tp8_pp2.json 16 128

The 405B TP8×PP2 band on ShareGPT-750 from the runner:

| Arm | req/s | Mean / p99 TTFT (ms) | Mean / p99 TPOT (ms) | Simulated |
|---|---|---|---|---|
| low | 5.84 | 7,785 / 19,498 | 60.4 / 71.0 | 128.3 s |
| nominal | 4.68 | 17,517 / 41,798 | 76.6 / 91.7 | 160.4 s |
| high | 3.90 | 27,361 / 64,143 | 92.5 / 112.8 | 192.4 s |

The runner uses upstream's scheduler defaults, so the fork's 405B latencies are
not comparable with its output. On the nominal arm:

- Upstream caps `--max-num-seqs` at 128; the fork had no cap. With
  `--max-num-seqs 1024 --no-enable-prefix-caching`, mean TTFT drops from
  17.5 s to 196 ms. Also passing `--no-enable-chunked-prefill` changes almost
  nothing (190 ms TTFT, 97.6 ms TPOT).
- With those settings matched, TPOT is still about a third below the fork's
  (97.6 vs 148.9 ms) and throughput is higher (6.02 vs 5.31 req/s). The likely
  cause is pipeline scheduling. The fork removed a request from its queue
  until its batch finished, so at PP=2 a request advanced once per full
  pipeline pass. Upstream follows vLLM and lets a request sit in both
  in-flight microbatches. That is inherited upstream behaviour, not a porting
  error, but no experiment has isolated it.

With the fork's settings (`--max-num-seqs 1024 --no-enable-prefix-caching`),
the migrated 405B TP8×PP2 band on ShareGPT-750, with the fork's own numbers in
parentheses:

| Arm | req/s | Mean / p99 TTFT (ms) | Mean / p99 TPOT (ms) |
|---|---|---|---|
| low | 6.81 (6.37) | 136 / 399 (231 / 693) | 68.4 / 88.6 (94.9 / 141.8) |
| nominal | 6.01 (5.31) | 196 / 528 (389 / 1,119) | 98.3 / 138.9 (148.9 / 263.8) |
| high | 5.23 (4.54) | 269 / 713 (649 / 1,862) | 131.6 / 199.2 (209.2 / 453.3) |

The arms stay in the same order. The band sits lower and is narrower: high/low
TPOT is 1.9x against the fork's 2.2x. These are calibrated projections, not
measurements.

## Validating a change

Compact controller mode now routes asynchronous ASTRA console diagnostics to
stderr, leaving stdout for READY/TEMPLATE/COMPLETE wire records. A captured
failure showed RingTopology logging inserted into the middle of READY, leaving
both processes waiting. Both analytical frontends use the isolated sink.
Malformed READY records and unexpected EOF now fail clearly rather than hang.
This fixes the reproduced compact-protocol corruption; arbitrary concurrent
runs in one checkout have not been certified safe.

File-free modes currently reject explicit `dp_group` configurations and any
non-analytical backend at startup. Independent replicas and TP/PP configurations
remain supported. Use legacy mode for explicit DP groups; this is a known
compatibility boundary, not a completed DP implementation.

    ./serving/validate.sh --clocks-only

58 scenarios against recorded clock counts, exact equality. A flag that only
changes *how* the graph reaches ASTRA-Sim must not move a single clock, so
those modes are checked against the same baselines:

    EXTRA_ARGS="--execution-template-mode shared-template --compact-controller-protocol --template-cache-max-entries 128" \
      PYTHON="$PWD/env/bin/python" ./serving/validate.sh single tp_pp multi_instance_8npu

A full shared-template invocation includes 12 intentionally rejected explicit-DP
scenarios. They must not be counted as clock-equivalence passes. Validate the
46 supported scenarios separately and test the startup rejection explicitly.

Unit tests (no GPU or vLLM required):

    ./env/bin/python -m unittest discover -s tests

2026-09-14 repair checkpoint: 42 unit tests pass; all 46 supported
shared-template clock scenarios match baselines after rebuilding both
analytical backends. Three real ShareGPT requests have identical legacy/shared
CSVs for 405B TP8×PP2 and 70B TP8/72 NPUs. Tier aggregate smoke passes.
The scale750 runner starts and progresses, but its bounded 180-second check
timed out (143) before completion; do not cite it as a full benchmark. Use a
longer serial run before reporting scale750 performance. Host-monitor executable
classification was also corrected to recognize AnalyticalAstra.

2026-09-15 checkpoint, after the host-memory fixes and a rebuild of both
analytical backends: 42 unit tests pass. `./serving/validate.sh --clocks-only`
matches 58/58 in legacy mode, and all 46 supported scenarios in shared-template
mode with the compact protocol and a 128-entry cache. The regenerated
`bench/examples` `sim.csv` and `summary.txt` files are byte-identical to the
committed ones. Stage 2 of `validate.sh` needs matplotlib, which `env/` lacks,
so it reports `Accuracy: FAIL` on the import alone. Run that stage as
`PYTHON=<a python with matplotlib> ./bench/examples/validate.sh`.
