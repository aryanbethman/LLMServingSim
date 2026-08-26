# LLMServingSim HiPC Context

_Last updated: 2026-08-26_

## Project boundary

This is the Marvell/HiPC topology-aware tiered-memory project. It is separate from
PriceKV/KV-eviction work. Do not merge or cherry-pick eviction code, configurations,
outputs, or results into this branch. Eviction is disabled for this project.

## Repository and access

- Remote repository: `/home/marvell/LLMServingSim` on `marvell@anjuna3.dashlab.in`.
- Branch: `feature/tiered-memory-topology`.
- Current branch head: `0dad641` (in-memory protocol plus 16-NPU config); the generic tier/fabric prototype entered at `c242cc5`.
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

The controller framing test, C++ parser harness, feeder equivalence harness,
tiered-memory unit suite, and analytical rebuild pass. This is not yet an
end-to-end simulation validation and it intentionally transports raw payloads.
The next protocol revision will transmit structural templates plus rank overlays.

An earlier shared-root 16-NPU run was stopped before it could interfere with the active PriceKV job, which uses the same generated ASTRA input tree. The isolated ShareGPT-750 validation then completed with exit 0, zero dynamic rank-ET workload files, and an exact match to the retained file-mode control: 750 requests and 89,673,372,165 total clocks. The live controller-to-memory-feeder handoff and raw in-memory ET transport are therefore behaviorally equivalent to the legacy file mode at this workload/scale. This does not validate the remaining structural-template or streamed-metrics work.

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
