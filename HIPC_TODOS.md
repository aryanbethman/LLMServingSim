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

## Active HiPC execution plan

- [x] **Establish representation correctness.** The 16-NPU legacy, shared,
      fused, and bounded-cache controls exactly match. The 72-NPU unbounded
      and bounded-128 pair also exactly match.
- [x] **Instrument host resources before 256 NPUs.** Added
      `analysis/monitor_simulation_resources.sh` (5-second process-tree RSS,
      CPU, FDs, result bytes/files, free disk/inodes),
      `analysis/write_run_manifest.py` (configuration/dataset/source hashes),
      and `analysis/run_scale_experiment.sh` (persistent runner). A 7-sample
      descendant-process smoke test passed at
      `/home/marvell/hipc-results/monitor-smoke-20260902`. `/usr/bin/time`
      remains supplementary because it records only the Python parent.
- [ ] **Primary simulator-scale sweep.** Fixed ShareGPT-750 rate-10 workload,
      Llama 3.1 70B/H100/FP16/TP=4, bounded cache 128, and logical-NPU ladder
      16 → 72 → 256 → 512 → 1,096. Preserve a two-hour cutoff and report a
      timeout as an operational limit. At 256+ use the bounded path only: the
      full unbounded/bounded exact comparisons at 16 and 72 already validate
      cache lifecycle correctness without doubling large-run cost.
- [ ] **Report simulator scaling, not TP scaling.** 256/512/1,096 mean 64/128/
      274 independent TP=4 replicas; do not claim one Llama model is sharded
      across hundreds of NPUs. Use wall time, ASTRA peak RSS, artifact count/
      bytes, template cache high-water/evictions, controller bytes, and request
      completion as the primary figures.
- [ ] **Separate later serving-capacity study.** Fixed ShareGPT-750 is suitable
      for simulator-host overhead but becomes lightly loaded at high replica
      counts. Any service-throughput claim requires proportionally scaled real
      ShareGPT arrivals/request repetitions and must be labeled separately.

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

- [~] **Primary scalability contribution:** replace ASTRA's rank-specific ET
      files with shared/content-addressed execution templates and streamed
      aggregate metrics. Cleanup only bounds peak retained storage; this work
      makes the 72/96/1,096 logical-NPU experiments credible.
  - [x] Measure rank-ET structural identity and converter/parse/filesystem cost
        on the validated 16-NPU ShareGPT-750 workload: 63,732 rank ETs / 14.23
        GB across 15,933 batches. Corrected profiling confirms 3,672 structural
        templates (94.2% upper-bound reuse), versus only 10.3% for exact payload
        hashes; no ET parse failures.
  - [x] Refactor Chakra conversion into a callable API that returns ET payloads;
        keep its current file-writing CLI as the legacy fallback. The new
        `convert_to_payloads()` reuses the legacy conversion/encoding path with
        `BytesIO` sinks. Byte-exact tests cover COLOCATED, DECODE, PREFILL, and
        EVENT; a real 70B batch also matched (four ranks, 893,042 bytes).
  - [~] Add a content-addressed template store and active-template lifetime.
        Implemented an opt-in ASTRA cache bound via
        `--template-cache-max-entries=N` (zero retains the compatibility,
        unbounded cache). Immutable template nodes are shared-owned by active
        rank feeders, so only map-only/inactive entries are reclaimed. ASTRA
        emits `TEMPLATE_RELEASE <sha256>` on eviction; the Python controller
        invalidates only that ID and retransmits it safely if later needed.
        Streamed metrics now include live/high-water cache entries/nodes,
        evictions, blocked evictions, and release notifications. Python
        protocol/template tests (7) and both analytical ASTRA backends build.
        Forced-small-cache (one entry) 16-NPU ShareGPT-750 validation completed
        at `/home/marvell/hipc-results/npu16-template-reclaim-20260901`: exit
        0, exact match to the final fused/deduplicated path (750 requests and
        89,673,372,165 simulated ns). It forced 8,983 evictions/releases;
        cache high-water was 21 templates / 23,625 nodes and final live cache
        was 16 / 18,000—bounded by active feeders rather than the 3,672
        historical templates. It took 11m34.70s / 263,560 KB, 9.2% slower
        than unbounded dedup due to deliberate retransmission, but 45.1% faster
        than the fresh legacy control. The resulting 652.15 MB transport is a
        bounded-cache stress case, not the preferred performance configuration.
  - [~] Extend ASTRA's ETFeeder/workload protocol to accept in-memory template
        payloads and instantiate small mutable per-rank execution state, rather
        than opening llm.<rank>.et files. The C++ ETFeeder now accepts an
        immutable shared byte payload and builds its own mutable dependency
        graph; it matched all 1,125 nodes of a real 70B ET. The experimental
        raw-payload controller/Workload path is implemented and unit/build-tested;
        an earlier shared-root run was stopped to protect the generated ASTRA input tree. The isolated 16-NPU ShareGPT-750 validation completed with exit 0, zero dynamic rank-ET workload files, and an exact match to the retained file-mode control (750 requests; 89,673,372,165 ns total clocks). The structural-template successor now sends each SHA-256-addressed template once and sparse per-rank overlays thereafter; ASTRA caches the immutable template and reconstructs the legacy ET for its existing mutable feeder. Python tests and an analytical rebuild pass; the isolated 16-NPU validation completed with exit 0, zero rank-ET workload files, and an exact 750-request/89,673,372,165-clock match to retained legacy mode. The direct structural successor is now implemented and builds: ASTRA parses template nodes once into an immutable cache, carries rank overlays separately, and has ETFeeder clone nodes directly into its mutable dependency graph without constructing a framed rank ET stream. Persistent direct-feeder and retained legacy controls both completed exit 0; the exact comparator matched all 750 requests, 89,673,372,165 simulated ns, and aggregate metrics. Direct feeder took 40m55.46s / 281,360 KB versus legacy 21m06.40s / 283,672 KB, so it is correctness-validated and filesystem-safe but not yet a wall-time win. Remaining: active-template reclamation and validation of trace-stream/compact-protocol changes.
  - [x] Shared-template mode is implemented; legacy remains the default and shared-template mode exactly matches the 16-NPU legacy control.
  - [~] Replace per-rank metric writes with a process-wide streaming aggregator
        and optional sampled per-request output. The first collection completed
        simulation but exposed an absolute-output-path bug; the fix is
        syntax-checked. Persistent collection completed: 31,866 bundles,
        29,937,662,780 wire bytes, 3,672 template definitions, 4,131,000
        template nodes, and 127,464 rank bindings.
  - [~] Remove the remaining per-batch text-trace round trip from non-legacy
        modes. Investigation during the 16-NPU retry found 17,823 generated
        trace files / 3.0 GB while the run was active, plus 21.7 million Python
        reads. Non-legacy trace synthesis now uses StringIO and passes finalized
        text directly to Chakra; the legacy file path remains unchanged.
        Analytical build/tests pass. The isolated persistent 16-NPU run (with
        compact controller protocol disabled) exactly matched all 750 legacy
        requests and 89,673,372,165 simulated ns, completing in 15m54.45s /
        277,468 KB versus the fresh legacy control's 21m06.40s / 283,672 KB:
        a 24.6% wall-time reduction. It also recorded zero filesystem input
        blocks; only final result/log output remains.
  - [~] Do not embed a complete text trace in every rank's GlobalMetadata for
        in-memory/template modes. ASTRA consumes only the ET nodes; use a small
        provenance digest instead and validate identical simulated results.
        Implemented with SHA-256; a representative rank ET shrank from 223,147
        to 48,680 bytes while all executable nodes matched. The same persistent
        trace-stream run exactly matched legacy output; this validates
        correctness jointly with in-memory trace synthesis, not an isolated
        timing contribution.
  - [~] Fuse converter output directly into the shared-template builder so a
        non-legacy batch does not serialize each rank's ET only to parse it
        back in Python. The candidate captures live protobuf nodes, applies the
        existing normalization/overlay logic, and produced a byte-identical
        bundle in an isolated component test. `--template-bundle-builder=payload|fused`
        now keeps payload splitting as the controlled default and enables
        fusion only for its own experiment. The fused 16-NPU run completed at
        `/home/marvell/hipc-results/npu16-fused-template-20260829`, with trace
        streaming, compact metadata, and compact controller protocol enabled.
        It exactly matches legacy and payload-builder output (750 requests /
        89,673,372,165 simulated ns) in 10m53.88s / 267,980 KB: 21.5% faster
        than compact payload mode (13m53.27s) and 48.4% faster than fresh
        legacy. Its transport counters report 7,344 template definitions /
        8.262M nodes / 539.35 MB versus payload mode's 3,672 / 4.131M /
        286.99 MB. Investigate apparent duplicate template accounting or
        transmission before making a large-scale transport claim. Fixed
        empty-cache reference handling and added controller counters for
        already-sent template IDs/nodes; that replay confirmed 3,672 real
        duplicate template sends / 4.131M duplicate nodes. A controller-side
        content-address filter now suppresses cached templates (unit-tested).
        Its end-to-end validation completed at
        `/home/marvell/hipc-results/npu16-fused-dedup-20260829`: exact compact
        payload-mode match, 10m36.12s / 266,252 KB, 23.7% faster than compact
        payload mode and 49.8% faster than legacy. Actual transmitted metrics
        are restored to 3,672 definitions / 4.131M nodes / 286.99 MB. The
        duplicate counters record suppressed upstream re-emissions, not bytes
        sent to ASTRA.
  - [~] Replace line-oriented controller polling and default ASTRA progress logs
        with a compact completion protocol and aggregate counters. Implemented
        as opt-in `--compact-controller-protocol`: ASTRA emits `READY rank
        iteration cycle exposed_cycles` and final `COMPLETE`/`INCOMPLETE`
        records; legacy remains default. Python parser and C++ build pass. The
        controlled 16-NPU run used `--template-bundle-builder=payload` to
        isolate this protocol change and exactly matched all 750 legacy
        requests / 89,673,372,165 simulated ns. It took 13m53.27s / 263,268 KB
        versus trace-stream's 15m54.45s / 277,468 KB and fresh legacy's
        21m06.40s / 283,672 KB: 12.7% faster than trace streaming alone and
        34.2% faster than legacy.
  - [~] Prove bounded file count/storage, template reuse, runtime, and peak RAM
        at 16, 72, 256, 512, and finally 1,096 logical NPUs. Added reproducible
        `analysis/generate_colocated_scale_configs.py`, generating colocated
        TP=4 baselines; extend it with the planned 256/512 points. The
        72-NPU config preflight verified ASTRA topology `[4,18]` and all 18
        controller/end-rank pairs. The normal fused/deduplicated 72-NPU
        reference completed exit 0 at
        `/home/marvell/hipc-results/npu72-fused-dedup-20260901`: 750 requests,
        zero dynamic artifacts, 3,244 templates / 3.6495M nodes, 372.33 MB
        transport, and 1h08m42s wall time. The paired 128-entry bounded-cache
        replay completed exit 0 at
        `/home/marvell/hipc-results/npu72-template-reclaim-128-20260901` and
        exactly matches all 750 reference results / 89,403,759,848 simulated
        ns. It held 128 cached templates (132 / 148,500-node high-water) and
        safely evicted/released 7,296 entries. It took 1h09m38s / 265,872 KB,
        only 1.4% slower than unbounded, while transport rose from 372.33 MB to
        658.80 MB due to intentional retransmission. Both runs emitted zero
        dynamic trace artifacts. Persistent ASTRA host monitoring is now
        implemented and recording a live 256-NPU bounded-128 run (64 TP=4
        replicas) at `/home/marvell/hipc-results/npu256-template-reclaim-128-20260902`.
        It persists `manifest.json` plus 5-second `host_resources.csv` samples
        with separate ASTRA/Python RSS, CPU, FDs, result bytes/files, and free
        disk/inodes. After this result, run 512 (128 replicas), then 1,096
        (274); omit 96 because it is too close to 72.
- [ ] Re-run the 16-NPU test after each simulator code change; after the
      validated bounded-cache path, use 72/256/512/1,096 for scale evaluation.

## Validation and case studies

- [x] Repeat the retained-trace run with interval artifact monitoring: exit 0,
      751 CSV lines, 23m 31s; final 79,665 artifacts / 17,081,453,333 bytes.
      The paired cleanup/retained artifact-growth plot can now be produced.

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
- [ ] Logical scale: 16, 72, 256, 512, 1,096 as TP=4 replicas.
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
### Upstream synchronization audit — 2026-09-02

- [x] Fetched `casys-kaist/LLMServingSim` and published its current
      `a4053bc` revision as `origin/sync/upstream-a4053bc`. The active
      `feature/tiered-memory-topology` checkout and its running 256-NPU
      experiment were deliberately left untouched.
- [~] Upstream is 129 commits ahead of our divergent `main` (which has two
      fork-only release commits), and changes the repository layout. It cannot
      be fast-forwarded or merged into the experiment branch safely; migration
      must be a later, separate port.
- [x] Novelty audit: upstream has in-process trace conversion, graph caching,
      compact metadata, and less idle ASTRA polling, but still materializes
      rank-specific ET files. Our shared/content-addressed structural ETs,
      rank overlays, bounded active-template reclamation, and large-scale host
      instrumentation remain non-overlapping. Do not claim generic
      "in-memory conversion" as the paper novelty.
- [x] Tried a pristine upstream 16-NPU H100/70B/TP=4/ShareGPT-750 control in
      `/home/marvell/LLMServingSim-upstream-a4053bc`. Its analytical ASTRA
      build succeeded, but upstream exits before ASTRA because its pinned
      Chakra `LLMConverter` lacks the `convert_rows` API called by
      `serving/core/graph_generator.py`. Do not patch this control: that would
      no longer measure upstream. The feature-branch 256-NPU run is paused,
      not discarded, pending a decision on a runnable comparison revision.
