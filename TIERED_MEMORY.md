# Topology-aware tiered-memory serving

This branch keeps the legacy simulator path unchanged unless a cluster configuration
declares `memory_tiers`. The new path is intended for P/D experiments with
eviction disabled.

## Generic configuration

A tier has a name, capacity, service bandwidth, base-access latency, sharing scope,
and a fabric endpoint:

```json
{
  "name": "cxl_pool",
  "capacity_gb": 512,
  "service_bw_gbps": 128,
  "access_latency_ns": 250,
  "sharing_scope": "rack",
  "endpoint": "cxl-switch"
}
```

Bandwidth values follow the simulator's existing convention: a value expressed in
GB/s is numerically equal to bytes/ns. Fabric links are directed and specify
`src`, `dst`, `bandwidth_gbps`, `latency_ns`, and a
`contention_group`. Links in the same group serialize traffic. Static routing is
the deterministic minimum-latency path over this directed graph.

Instances participating in P/D set `kv_tier`. On prefill admission, the simulator
reserves its block-rounded KV capacity. At handoff it releases the source
reservation, reserves the destination, transfers 1/4/16-block chunks (configured
by `pd_transfer.chunk_blocks`), and holds decode until the configured prefetch
lookahead is ready.

The transfer model charges source-tier read service, every fabric hop, contention,
and destination-tier write service. It records block precision in bytes, ownership,
completion time, reservation failures, prefetch-ready blocks, admission wait,
tier occupancy, and link bytes/busy time/utilization.

## Example

`cluster_config/tiered_memory_pd_h100_tp4.json` is an H100 TP=4 P/D skeleton
with local HBM, host DRAM, switched CXL, and remote-HBM tiers. It uses a two-hop
prefill-to-decode route; the bandwidth and latency values are calibration inputs,
not a claim of NVL72 fidelity. To compare a destination tier, copy the file and
change the decode instance's `kv_tier` to `host_dram`, `cxl_pool`, or
`remote_hbm`; the directed links in the example include paths to each endpoint.

Run a full workload with the project runtime:

```sh
PATH=/home/marvell/LLMServingSim/env/bin:$PATH \
/home/marvell/LLMServingSim/env/bin/python3 main.py \
  --cluster-config cluster_config/tiered_memory_pd_h100_tp4.json \
  --dataset dataset/sharegpt_req750_rate10_llama.jsonl --num-req 750 \
  --tier-stats-output output/tiered_memory/sharegpt750.json
```

Generated dynamic trace/workload artifacts are removed once all ranks finish a
batch, which bounds peak disk/inode growth. Pass `--retain-traces` to keep them
for debugging. Converter output remains rank-specific because ASTRA-Sim consumes
per-rank ET files; content-addressed template reuse and in-memory rank
instantiation are not yet implemented.

## Evaluation matrix

Use only the raw ShareGPT-750, -1000, and -1500 datasets in
`dataset/TIERED_MEMORY_WORKLOADS.md`.

- Models: Llama 3.1 8B (TP=1), Llama 3.1 70B (TP=4), Mixtral-8x7B (TP=4).
- Tiers: local HBM, host DRAM, switched CXL, remote accelerator HBM.
- Paths: local, one switch, two-hop pool; oversubscription: 1:1, 2:1, 4:1.
- Logical scale: 4, 16, 32, 72, 96, and 1,096 NPUs as TP<=4 replicas.
- Sweep chunk size: 1, 4, 16 blocks; prefetch lookahead: 0, 2, 8 blocks.

The already-completed pre-change 16-NPU ShareGPT-750 baseline is the only
baseline retained. Do not launch 96- or 1,096-NPU baseline runs unless directed.

## Validation status

The cleanup path reduces retained trace/workload artifacts by deleting them after
the scheduler reports a batch complete. It must not yet be treated as correctness
validated: the current scheduler acknowledgement observes the first and last NPU,
not an explicit all-rank ET-file lifetime acknowledgement. A retained-artifact
control run and exact request/metric comparison are required before using
cleanup-enabled measurements in the paper. See `llmservingsim_context.md` for the
current run state and hand-off details.
