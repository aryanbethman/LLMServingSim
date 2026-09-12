# LLMServingSim HiPC Context

_Last updated: 2026-09-09_

## Current Phase 1–2 profile status

The active feature branch now contains a deterministic profile-projection path,
separate from topology/eviction work:

- `llm_profile/profile_projection.py` reconstructs H100/Llama-3.1-70B TP=8
  from checked-in TP=1/2/4 measurements and writes manifests with source hashes,
  source/Git hashes, method version, and explicit measurement status.
- `tp8_v0_reference/` freezes the historic TP=8 profile.  Its generated v1
  successor matches all layer and decode-attention rows semantically, and
  prefill rows through 2,048 tokens.  The historic long-prefill rule was not
  encoded, so changed >2,048-token rows are recorded in `v0_comparison.json`;
  v0 remains canonical for compatibility.
- The held-out TP1/2→TP4 back-test is 3.89% median / 21.78% P90 absolute error,
  satisfying the 10%/25% gates.  CSV/PNG evidence lives in the v1
  `calibration/` directory.
- `model_config/meta-llama/Llama-3.1-405B.json` and the generated H100/405B
  TP=8 nominal/low/high profiles are calibrated projections, not measurements.
  They use 126 layers, hidden 16,384, intermediate 53,248, 128 Q heads, 8 KV
  heads, BF16, and 131,072 context.
- The 405B memory calculator assumes TP=8×PP=2 (16 H100-80GB GPUs): 50.73 GB
  weights/GPU and 32,256 KV bytes/token/GPU.  TP=8 alone is intentionally
  capacity-infeasible; PP=2 is required before a full serving run.
- Five profile-projection tests and the 14 existing tiered-memory tests pass.
  A direct in-memory trace-generation smoke test for 405B/TP=8 passed; it
  exercises profile lookup without trying to admit an impossible TP=8-only run.
  `llm_profile/PROFILE_PROJECTIONS.md` is the reproduction guide.
- An 8-logical-NPU control using one request drawn from raw ShareGPT-750 confirms
  profile/execution compatibility: legacy and fused shared-template results
  match exactly (1,580,162,459 simulated ns).  The shared path emitted 32
  templates/1,808 rank bindings and zero dynamic workload directories.  Invoke
  full simulations with `PATH=/home/marvell/LLMServingSim/env/bin:$PATH` so
  graph-generator child Python processes can import Chakra.
- **405B is now end-to-end runnable** in the initial 16-logical-H100
  configuration (`cluster_config/llama_405b_h100_tp8_pp2.json`).  It uses TP=8
  × PP=2: two contiguous 63-block stages.  Memory admission reserves the largest
  stage, while KV capacity is accounted per stage.  Chakra now consumes explicit
  transformer-block boundaries rather than splitting a flat operator list: rank
  0 sends after `down_proj_756`; rank 8 starts at `input_layernorm_757`.
  One raw ShareGPT-750 request completed in 6,092,460,472 simulated ns; legacy
  and shared-template results match exactly. The full nominal ShareGPT-750
  workload now also completed: 750/750 records, 3m47.040s simulated time,
  5.31 req/s, mean/p99 TTFT 388.54/1,118.57 ms, and mean/p99 TPOT
  148.88/263.79 ms. Its persistent result directory is
  /home/marvell/hipc-results/llama405b-h100-tp8-pp2-sharegpt750-nominal-retry1-20260909/
  (2.3 MB retained). A separately monitored nominal replay completed in
  3m51.12s wall-clock, with 3m47.040s simulated time, 275,628 KB
  Python-parent RSS, and 6,323,644 KB peak process-tree RSS. These are
  calibrated-profile results, not physical 405B/H100 measurements.
- Projected attention tables have finite batch/KV grids.  Missing lookup points
  now use a cached interpolation/edge-extrapolation fallback while exact rows
  remain unchanged.  This is required for valid high-concurrency 405B batches
  beyond the source TP=4 table's batch-256 limit and is explicitly a projection.
  The explicit --profile-variant nominal|low|high command-line selector resolves
  immutable profile directories and includes the variant in every in-process
  profile-cache key; uncertainty runs therefore cannot contaminate nominal data.
  The low/high ShareGPT-750 sensitivity pair is complete. All three variants
  completed 750 requests and created zero dynamic workload directories. Low /
  nominal / high: throughput 6.37 / 5.31 / 4.54 req/s; mean TTFT 230.81 /
  388.54 / 648.61 ms; p99 TTFT 692.76 / 1,118.57 / 1,861.73 ms; mean TPOT
  94.93 / 148.88 / 209.23 ms; p99 TPOT 141.81 / 263.79 / 453.33 ms. This is
  an uncertainty band for a calibrated projection, not physical measurement.
  Do not begin ShareGPT-1000/-1500 until requested.

## Current topology-aware tier status

The branch now has an end-to-end P/D tier path for named HBM, host-DRAM, CXL,
and remote-HBM tiers. It reserves source/destination capacity, routes each KV
handoff over directed links with contention groups, and emits tier/link metrics.
Decode attention now adds only the extra selected-tier KV-read service and return
fabric cost beyond its measured local-HBM baseline. The two-request H100 TP4
CXL smoke completed with 379 reads, 6.23 GB CXL read traffic, and 95.68 ms
aggregate added read latency; the matched local-HBM smoke remains unchanged.
This validates the mechanism, not an externally calibrated CXL system.

## Project boundary

This is the Marvell/HiPC topology-aware tiered-memory project. It is separate from
unrelated prior work. This branch is reserved for topology-aware tiered-memory simulator development.
outputs, or results into this branch. Eviction is disabled for this project.

## Repository and access

- Remote repository: `/home/marvell/LLMServingSim` on `marvell@anjuna3.dashlab.in`.
- Branch: `feature/tiered-memory-topology`.
- Current branch: feature/tiered-memory-topology; consult git log for the moving implementation head. The generic tier/fabric prototype entered at c242cc5.
- The local project runtime is:
  `/home/marvell/LLMServingSim/env/bin/python3`.
- Off-campus access uses:
  `ssh -J dashlab@campnet.dashlab.in,dashlab@lab.dashlab.in marvell@anjuna3.dashlab.in`.
  Key-based access is configured; do not record credentials in this file.
- ASTRA-Sim remains unchanged except for the scoped nested Chakra converter
  update described below; do not alter unrelated ASTRA code or generated inputs.

## Implemented prototype

The branch adds a generic tier/fabric path while preserving legacy configurations
when no `memory_tiers` section is provided.

- `inference_serving/tiered_memory.py`: capacity-aware memory tiers; directed
  links; deterministic static minimum-latency routing; per-link contention;
  source read, fabric traversal, destination write; tier/link metrics.
- `memory_tiers` configuration fields: capacity, service bandwidth, base latency,
  sharing scope, endpoint.
- `fabric.links` fields: source, destination, bandwidth, latency, contention
  group.
- P/D KV handoff state: block ownership, precision, readiness, completion,
  reservation, 1/4/16-block chunks, and prefetch controls.
- `cluster_config/tiered_memory_pd_h100_tp4.json`: H100 TP=4 P/D example with
  local HBM, host DRAM, CXL pool, remote HBM, and directed paths.
- `--tier-stats-output`: exports transfer bytes/stalls, prefetch/admission
  information, link busy time/utilization, and tier occupancy.
- `--cleanup-consumed-traces`: explicit experimental cleanup; dynamic artifacts
  are retained by default. `--retain-traces` remains a compatibility no-op.

## Trace-artifact scalability change

Previous behavior retained every generated per-batch trace and Chakra workload
directory. Cleanup was behaviorally validated against a retained-artifact control:
all 750 request rows and tracked simulated metrics matched exactly.

The ASTRA audit found that Python sees completion reports only from controller/end
ranks. Managed ranks consume their ET files internally, so their lifetime is not
explicitly acknowledged to Python. Cleanup is therefore deliberately explicit
opt-in via `--cleanup-consumed-traces`; the default retains artifacts. This avoids
claiming a stronger filesystem-lifetime guarantee than the current ASTRA interface
provides.

A run-start bug caused by reusing `args` for the ASTRA subprocess command was fixed
in `main.py`: trace-policy and tier-stat arguments are captured before that reuse
(commit `67fbfe5`).

## In-memory converter API

The first shared-template prerequisite is implemented in the nested Chakra
submodule (commit `52f8155`, pinned through ASTRA commit `9c87b60`).
`LLMConverter.convert_to_payloads()` returns rank-indexed ET byte payloads
without creating rank files. Its legacy `convert()` file-writing interface
remains the default and uses the same internal conversion/encoding path.

Automated byte-exact equivalence tests cover COLOCATED, DECODE, PREFILL, and
EVENT inputs; a real Llama 3.1 70B batch also matched for four ranks (893,042
bytes). This is only the producer-side API: no ASTRA ETFeeder or simulator
execution path consumes in-memory templates yet.

## In-memory feeder API

The nested Chakra feeder now accepts an immutable shared ET byte payload
(commit `74b3ce3`, pinned through ASTRA commit `dfd1d38`). Each ETFeeder creates
an independent mutable dependency graph, so ranks do not share node state.
An analytical rebuild succeeded, and a standalone comparison against a real
Llama 3.1 70B ET matched all 1,125 issued nodes.

This is a consumer-side prerequisite only. The controller still passes
rank-file paths across stdin and Workload still selects the file constructor.
The next integration slice is a framed template-bundle protocol over the
existing controller pipe, followed by legacy/shared-mode equivalence tests.

## Raw ET controller protocol

The experimental raw-payload protocol is complete in source (ASTRA commit `b69d0d1`). Python sends an
`ET_PAYLOADS` JSON/base64 command over the existing stdin controller
pipe. The analytical frontend decodes rank ET bytes, and Workload uses the
in-memory ETFeeder rather than rank-file paths. The runtime selector is
`--execution-template-mode in-memory` (analytical backend only); legacy
file mode remains default.

The controller framing test, C++ parser harness, feeder equivalence harness, tiered-memory unit suite, and analytical rebuild pass. Raw in-memory and shared-template transport have both passed complete 16-NPU end-to-end comparisons.
The shared-template protocol is now implemented: controller sends SHA-256-addressed structural templates once and sparse rank overlays thereafter; ASTRA caches templates and reconstructs rank ET streams for the existing feeder. Aggregate transport metrics can be emitted as one JSON summary. The isolated 16-NPU validation completed with an exact legacy match and zero rank-ET files.

The first shared-root 16-NPU run was stopped to prevent interference through the generated ASTRA input tree. The isolated ShareGPT-750 validation then completed with exit 0, zero dynamic rank-ET workload files, and an exact match to the retained file-mode control: 750 requests and 89,673,372,165 total clocks. The live controller-to-memory-feeder handoff and raw in-memory ET transport are therefore behaviorally equivalent to the legacy file mode at this workload/scale. This does not validate the remaining structural-template or streamed-metrics work.

## Workloads

Only these raw datasets were copied from `aryan/dev0`; hashes are recorded in
`dataset/TIERED_MEMORY_WORKLOADS.md`.

- ShareGPT-750
- ShareGPT-1000
- ShareGPT-1500

## Scalability baseline status

The only completed pre-change baseline is unmodified `main`:

- Workload: ShareGPT-750, 750 requests.
- Model: Llama 3.1 70B; H100 profile; TP=4.
- Layout: four colocated replicas = 16 NPUs.
- Completion: success.
- Wall time: 23m 09.75s.
- Maximum root-process RSS: 273,308 kB.
- Filesystem outputs: 39,016,744.
- Dynamic trace/workload artifacts: 79,686 files; ~16.7 GB retained.
- Results directory: `/tmp/llmservingsim-baseline-results/npu16-run2`.

Do not run 96- or 1,096-NPU pre-change baselines.

The cleanup-enabled post-change 16-NPU ShareGPT-750 run completed successfully:

- Worktree: /tmp/llmservingsim-tiered-16.
- Results: /tmp/llmservingsim-tiered-results/npu16-postchange.
- Exit status: 0; 750 completed requests; 23m 14.10s wall time.
- Peak dynamic artifacts: 20 files / 4,299,676 bytes.
- Retained after completion: 0 files / 8,192 bytes directory overhead.
- This is a diagnostic result only until paired correctness validation passes.

The matched retained-trace control is active at commit 67fbfe5:

- Worktree: /tmp/llmservingsim-tiered-16-retained.
- Results: /tmp/llmservingsim-tiered-results/npu16-retained.
- Its sole intended behavioral difference is --retain-traces.
- analysis/compare_simulation_runs.py will compare request CSV rows and final
  simulated metrics once it completes.

## Revised HiPC paper plan

**Working title:** **Topology-Aware Simulation of Tiered-Memory LLM Inference Serving**

This is a simulator paper. Its question is: how should an LLM-serving simulator
represent non-uniform memory tiers, physical fabric paths, and P/D KV movement?

The central claim is methodological: topology is part of the memory model. A
uniform-link, fixed-remote-memory abstraction cannot expose placement-dependent
contention, transfer timing, or tail latency.

The contribution is the simulator abstraction and its validation:

1. Configurable memory tiers: capacity, service bandwidth, access latency, sharing
   scope, and fabric endpoint.
2. A directed fabric graph with static routing and contention groups.
3. Block-granular P/D KV ownership and movement: source read, every hop,
   destination write, readiness, chunks, and prefetch.
4. A scalable execution/metric path, subject to the trace-lifetime validation
   recorded below.
5. NVL72-like 18-tray case studies as calibrated projections, not TP=72 or
   measured-NVL72 claims.

Case studies demonstrate sensitivity of the simulator to tier, path, sharing, and
placement. They do not introduce or evaluate a new placement policy.

Out of scope: eviction, KV-value analysis, new serving scheduling algorithms, or
claims beyond TP<=4 ASTRA profile support. The detailed checklist is in
HIPC_TODOS.md.

## Evaluation scope

- Models: Llama 3.1 8B (TP=1), Llama 3.1 70B (TP=4), Mixtral-8x7B (TP=4).
- Workloads: ShareGPT-750/1000/1500 only.
- Tiers: local HBM, host DRAM, switched CXL, remote accelerator HBM.
- Paths: local, one-switch, two-hop pool.
- Sharing: 1:1, 2:1, 4:1.
- Blocks/lookahead: 1/4/16 and 0/2/8.
- Scale: 4, 16, 32, 72, 96, 1,096 logical NPUs as TP<=4 replicas.
- Metrics: throughput, TTFT, TPOT, p99 latency, decode admission stall, prefetch
  coverage, link/pool use, tier occupancy, trace file count, runtime, peak memory.

## Known limitations and priorities

1. ASTRA collectives remain FullyConnected and TP profiles only extend to TP=4.
2. The current cleanup solution bounds retained disk/inode use but does not implement
   shared execution templates or in-memory rank instantiation. That is the main
   remaining scalability contribution.
3. Generic CXL/remote inputs require public-spec calibration and 0.5x/1x/2x
   sensitivity analysis; label NVL72 results as calibrated projections.
4. A small H100 validation point is required before extrapolation.
5. Do not add eviction/recompression to this paper.

## Retained-control validation result

The retained-trace control completed. Exact comparison result: **MATCH**. Details: /tmp/llmservingsim-tiered-results/npu16-validation-comparison.log.

The interval-monitored retained control also completed successfully: exit 0,
23m 31s, 751 CSV lines, and 79,665 final artifacts /
17,081,453,333 bytes. Its monitor.tsv supplies the retained growth curve for
the cleanup-versus-retained presentation plot.
