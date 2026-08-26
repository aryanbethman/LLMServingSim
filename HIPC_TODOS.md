# HiPC Simulator Paper TODOs

## Paper scope

Working title: Topology-Aware Simulation of Tiered-Memory LLM Inference Serving.

This is a simulator paper. The central claim is that topology belongs in the memory
model for tiered-memory inference serving; uniform-link and fixed-remote-memory
abstractions cannot represent placement-dependent contention or tail latency.

Out of scope: eviction algorithms, KV-value/data-science analysis, a new serving
scheduler, TP=72 fidelity, or measured NVL72 performance.

## Immediate: correctness and reproducibility

- [x] Complete the cleanup-enabled 16-NPU ShareGPT-750 run (exit 0).
- [x] Record paired-run telemetry: cleanup exit 0, 750 requests, 23m 14.10s,
      20 peak artifacts / 4,299,676 bytes; retained exit 0, 750 requests,
      23m 30.84s, 79,665 artifacts / 17,081,453,333 bytes.
- [x] Run commit 67fbfe5 with --retain-traces (completed; see validation log).
- [x] Add analysis/compare_simulation_runs.py for exact CSV and metric comparison.
- [x] Run the exact comparator after the retained control completes.
- [x] Audit ASTRA's ET-file lifecycle: Python sees controller/end-rank reports;
      managed ranks consume ET files internally.
- [x] Make artifact cleanup explicit opt-in (`--cleanup-consumed-traces`) until
      ASTRA exposes an all-rank consumption acknowledgement.
- [x] Validate trace-policy CLI mutual exclusion and tiered-memory unit suite (7 tests).
- [x] Commit the main.py arguments-shadowing fix (67fbfe5).

## Simulator model

- [ ] Specify and test a legacy-to-generic configuration adapter without changing
      legacy run outputs.
- [ ] Add config validation for tier endpoints, directed routes, contention groups,
      sharing scope, and P/D source/destination tiers.
- [ ] Test block ownership, reservation/release, source read, destination write,
      one-hop/two-hop timing, and contention at integration level.
- [ ] Decide/document zero-prefetch and partial-KV semantics; do not claim
      compute/transfer overlap without modeling it.
- [ ] Export a stable metrics schema for tier occupancy, transfer stalls, link
      bytes/busy time, and artifact statistics.

## Scalability engineering

- [ ] **Primary scalability contribution:** replace ASTRA's rank-specific ET
      files with shared/content-addressed execution templates and streamed
      aggregate metrics. Cleanup only bounds peak retained storage; this work
      makes the 72/96/1,096 logical-NPU experiments credible.
- [ ] Profile trace generation, Chakra conversion, ASTRA startup, filesystem
      operations, and Python memory on the validated 16-NPU workload.
- [ ] Implement in-memory rank instantiation from shared templates.
- [ ] Preserve optional sampled per-request output without per-rank metrics files.
- [ ] Demonstrate bounded file count/storage and practical runtime at 72, 96,
      and 1,096 logical NPUs before presenting those scale results.
- [ ] Re-run the 16-NPU test after each scalability change; only then evaluate
      72/96/1,096 logical-NPU scenarios.

## Validation and case studies

- [~] Repeat the retained-trace run with interval artifact monitoring, so cleanup
      and retained storage growth can be plotted on the same time axis.
      Running in an isolated worktree with the validated ASTRA analytical build.

- [ ] Collect authoritative calibration inputs for H100 HBM, host DRAM, CXL, and
      remote-HBM/silicon-photonics links.
- [ ] Build 0.5x/1x/2x sensitivity configurations for uncertain inputs.
- [ ] Obtain one small physical H100 reference point, if available.
- [ ] Validate against it and describe remaining discrepancy.
- [ ] Implement an 18-tray, four-GPU-per-tray NVL72-like placement generator:
      TP=4 within tray; placement/fabric only across trays.
- [ ] Build local, one-switch, and two-hop path cases; 1:1, 2:1, 4:1 sharing.

## Evaluation

- [ ] Models: Llama 3.1 8B TP=1, Llama 3.1 70B TP=4, Mixtral-8x7B TP=4.
- [ ] Workloads only: ShareGPT-750, ShareGPT-1000, ShareGPT-1500.
- [ ] Tiers: local HBM, host DRAM, switched CXL, remote HBM.
- [ ] Sweeps: 1/4/16 block chunks and 0/2/8 lookahead.
- [ ] Logical scale: 4, 16, 32, 72, 96, 1,096 TP<=4 replicas.
- [ ] Report: throughput, TTFT, TPOT, p99, admission stalls, prefetch coverage,
      tier occupancy, link utilization, artifact count, runtime, and peak memory.

## Paper and poster

- [ ] Write the simulator model and assumptions section.
- [ ] Add a validation section before presenting projected case studies.
- [ ] Use case studies to demonstrate simulator sensitivity; label NVL72 outcomes
      as calibrated projections.
- [ ] Poster figures: configuration/model diagram; transfer timing path; validation
      plot; topology/tier sensitivity heatmap; scalability chart.
- [ ] State limitations prominently: TP<=4 profiles, FullyConnected collectives,
      uncertain remote-memory inputs, and no physical NVL72 validation.
