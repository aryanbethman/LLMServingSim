# LLMServingSim Marvell/HiPC handoff

## Purpose and status

This branch is the isolated Marvell/HiPC simulator project.

- Repository: github.com/aryanbethman/LLMServingSim
- Branch: feature/tiered-memory-topology
- Top-level baseline at handoff: current branch HEAD; ASTRA fork commit 76093de; Chakra fork commit 246b16d
- Scope: scalable, topology-aware, tiered-memory LLM-serving simulation.
- Out of scope: PriceKV, eviction policy research, and KV-value/data-science work.

Start with this file, then read the documents below in order. Do not infer that a
calibrated hardware projection is a physical measurement.

## Reading order

1. llmservingsim_context.md  current technical state, implemented behavior,
   experiment evidence, limitations, and paper plan.
2. HIPC_TODOS.md  completed work and exact remaining tasks.
3. llm_profile/PROFILE_PROJECTIONS.md  reproducible TP=8 and 405B profile
   methodology and boundaries.
4. TP8_PROFILE.md  deeper rationale, per-operator caveats, and the TP=8
   measurement-validation path.
5. cluster_config/llama_405b_h100_tp8_pp2.json  runnable 405B deployment.
6. cluster_config/tiered_memory_pd_h100_tp4.json and
   cluster_config/tiered_memory_pd_cxl_h100_tp4.json  local-HBM and CXL
   topology/tier smoke configurations.

## What is implemented

### Simulator scalability

- Legacy ASTRA rank-specific execution traces were replaced for non-legacy runs
  by shared, content-addressed structural templates plus sparse per-rank
  overlays.
- Direct/in-memory construction and compact controller transport reduce
  filesystem traffic while preserving legacy request outcomes.
- Bounded active-template caching, aggregate transport metrics, resource
  monitoring, and retained-vs-shared exact comparisons are present.
- The 16-NPU ShareGPT-750/H100/70B TP=4 controls exactly match legacy output.

### Reproducible TP=8 and 405B profiles

- H100/Llama 3.1 70B TP=8 is a versioned projection from measured TP=1/2/4
  data. The held-out TP=4 back-test is 3.89% median and 21.78% P90 absolute
  error.
- Llama 3.1 405B BF16 is geometry-aware: its actual architecture drives
  operator FLOPs, HBM bytes, activations, KV bytes, and TP payloads.
- The runnable deployment is H100 TP=8 x PP=2: 16 GPUs, two contiguous
  63-transformer-block stages. TP=8 without PP is deliberately rejected as
  memory-infeasible.
- Low/nominal/high 405B profiles are sensitivity variants. They are calibrated
  projections, not measured 405B/H100 performance.
- The 405B nominal ShareGPT-750 run completed 750/750 requests. Its monitored
  rerun took 3m51.12s wall-clock and 3m47.040s simulated time, with 6.32 GB
  peak process-tree RSS.

### Topology-aware memory

- Named HBM, host-DRAM, CXL-pool, and remote-HBM tiers have capacity, service
  bandwidth, base latency, endpoint, and sharing scope.
- Directed fabric links have bandwidth, latency, static shortest-latency
  routing, and contention groups.
- P/D handoff reserves capacity, models source read, fabric hops, destination
  write, and block/chunk transport.
- Decode from a non-local tier adds the tier-service plus return-fabric cost
  beyond the local-HBM cost already contained in the measured profile.
- Local-HBM and CXL P/D smoke tests pass. The CXL mechanism test records
  remote-KV read/link traffic and added decode latency.

## Evidence and limits

Keep these separate:

1. Simulator correctness: exact legacy vs shared-template request outcomes
   validate that scalability changes did not alter simulated behavior.
2. Profile-method accuracy: TP=1/2 to held-out TP=4 back-test validates the
   TP projection method.
3. Physical hardware accuracy: not yet available for exact 405B/H100 or future
   CXL/NVL72 systems. Results must be labeled calibrated projections and include
   sensitivity analysis.

Known gaps before making strong topology claims:

- configuration validation for every endpoint, route, contention group, and
  P/D source/destination combination;
- a fully explicit block-readiness/stall model for advanced partial-KV
  chunking/prefetch schemes;
- public/vendor calibration inputs for CXL, remote HBM, and photonic links;
- physical validation, where accessible.

## Reproduction essentials

Run from /home/marvell/LLMServingSim and prepend the project virtual environment
to PATH.

    export PATH=/home/marvell/LLMServingSim/env/bin:$PATH
    python -m unittest tests.test_pipeline_parallelism tests.test_profile_projection -v
    python -m unittest discover -s tests/tiered_memory -v

The 405B nominal full run uses:

    python main.py --cluster-config cluster_config/llama_405b_h100_tp8_pp2.json       --profile-variant nominal --fp 16 --block-size 16       --dataset dataset/sharegpt_req750_rate10_llama.jsonl --num-req 750       --network-backend analytical --execution-template-mode shared-template       --template-bundle-builder fused --compact-controller-protocol

Persistent experiment outputs are deliberately outside Git under
/home/marvell/hipc-results/. Do not use /tmp for new persistent results.

## Working-tree caution

Do not reset, checkout, delete, or commit unrelated working-tree changes without
the owner's approval. The checkout may contain a dirty ASTRA submodule and
untracked baseline configuration files from separate work. The tracked branch
content at the stated commit is the reviewable project state.
