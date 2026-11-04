---
sidebar_position: 4
title: Scaling Agent Sessions
---

# Scaling Agent Sessions

The session scale study measures the simulator host's cost while increasing
the workload offered to independent model replicas. Logical GPUs are simulated;
they are not physical GPUs executing inference on the simulator host.

## Preserve the workload, change the offered load

Generate a source corpus with the SWE-bench generator described in
[Agentic Sessions](./agentic-sessions.md), or use a recorded session JSONL.
The source must contain complete `sub_requests` chains and unique session IDs.

```bash
python -m workloads.generators replicate-sessions \
  --source outputs/source-sessions.jsonl \
  --output outputs/scaled-sessions.jsonl \
  --num-sessions 800 --sps 3.2 --seed 42
```

This preserves each selected session's calls, input/output lengths, tool
delays and token-prefix relationships. Only session identities, session start
times and the explicitly selected token policy change. Calls remain closed
loop: a later call is released after its predecessor finishes plus the
recorded tool delay. Increasing session count extends or enlarges the workload;
increasing arrival rate raises concurrency. Neither guarantees a busy cluster.

The generator samples shuffled passes over the source corpus. Repeated passes
are workload replication, not additional independent recorded tasks. Its JSON
manifest records the source/output SHA-256, seed, arrival spec, source-session
mapping and token policy. Existing outputs are never overwritten.

### Token identity and cache fidelity

- `--token-policy isolate-copies` gives each repeated corpus a disjoint token-ID
  range. Within one copy, all token equalities are preserved, including shared
  system prefixes between different source tasks and history reuse within a
  session. Separate copies cannot create extra cross-copy prefix-cache hits.
- `--token-policy preserve` keeps token IDs unchanged and deliberately models
  repeated identical content. It may increase cache reuse as the corpus repeats.

Relabeled IDs are synthetic cache-hash identifiers, not tokenizer vocabulary
IDs for running a language model. Token lengths and modeled layer work are
unchanged. Neither policy turns repetitions into fresh natural-language tasks.

The existing router selects an instance for every newly released call. Its
load policy considers waiting/running request counts, not session or KV-cache
affinity. This study keeps routing unchanged. An execution-template cache hit
accelerates the simulator; a KV-prefix cache hit changes the modeled serving
work. Report them separately.

## Prepare and run a matched experiment

Use a base configuration with the intended hardware and TP degree. Preparation
copies its first colocated instance and preserves its memory and link settings.
It does not infer new hardware values from a GPU name.

```bash
python analysis/session_scale.py prepare \
  --base-config experiments/session_scale/qwen_tp4_base.json \
  --source outputs/source-sessions.jsonl \
  --output-dir outputs/scale256 \
  --npus 256 --num-sessions 800 --sps 3.2 --seed 42

python analysis/session_scale.py run \
  --config outputs/scale256/cluster.json \
  --dataset outputs/scale256/sessions.jsonl \
  --result-dir outputs/scale256-shared-fast \
  --mode shared-fast --profile-variant bf16-tp4proj \
  --window-start 50 --window-end 200
```

Window values above illustrate the command. Choose and freeze a window from
calibration measurements; exclude warm-up and final drain. Do not choose a
window only because it makes the result look busy.

The Qwen TP4 profile is projected from measured lower TP degrees. It supports
relative simulator-efficiency comparisons on the same profile, not a validated
physical-cluster capacity claim.

To bound repeated timing cost, select a fixed random subset of complete chains
before running any timings. This command records the source/output checksums
and selected identities, and does not truncate calls:

```bash
python analysis/session_scale.py select-source \
  --source outputs/source-sessions.jsonl \
  --output outputs/timing-source.jsonl --num-sessions 6 --seed 42
```

Use this same selected corpus for both cluster sizes. The initial matched
design uses 6 sessions at 0.2 sessions/s on 16 GPUs, and 96 sessions at 3.2
sessions/s on 256 GPUs. Both population and arrival rate scale by sixteen.
This controls source mix and offered load per replica; calibration must still
establish whether the measurement window is busy. Report the selected corpus
and the larger all-source calibration separately.

The runner is Linux-only (`/proc`, GNU time). It refuses existing result
directories and concurrent invocations against the same checkout. It records
commands, source/input hashes, Git revisions/status, environment, process-tree
RSS and resource limits. Runs exceeding the configured timeout, disk floor or
memory limit are marked failed; only that invocation's UUID scratch directory
is removed. Failed runs are not accepted as complete comparisons.

For workloads whose legacy files exceed the local disk budget, pass
`--scratch-parent /path/to/large/scratch` to every matched run. Record the
filesystem: timings from local storage and network storage are different
experiments. All modes retain inputs through the timed simulation. The runner
then counts ET files and logical bytes and removes its isolated directory;
artifact counting and cleanup times are reported separately. GNU-time block
counts are host I/O accounting and may not represent network-filesystem traffic.

## Modes and the question each comparison answers

| Mode | Python graph production | ASTRA-Sim input |
| --- | --- | --- |
| `legacy` | Existing conversion and exact-repeat cache | Complete per-rank ET files |
| `legacy-fast` | Cached mutable per-rank protobuf nodes; direct value updates | Complete per-rank ET files |
| `overlay` | Normal conversion; structural templates and value lists | Shared templates and values |
| `shared-fast` | Learned row-to-value mapping skips conversion | Shared templates and values |

The runner uses the same compact controller protocol in every mode. `overlay`
disables the value fast path; `shared-fast` enables it. The mutable legacy
cache uses the same shape key and tagged mapping checks as the shared fast
path. It preserves per-rank serialization, writes and ASTRA parsing, rather
than rebuilding shared templates and then intentionally materializing them
through an expensive round trip.

`legacy-fast` versus `shared-fast` isolates the additional benefit of the
shared representation and delivery after both avoid graph construction.
`legacy` versus `legacy-fast` measures the gain available while retaining
files. `overlay` versus `shared-fast` measures the shared fast path.

For a stricter attribution experiment, add `--disable-repeat-caches` to both
matched runs. It bypasses legacy's exact-ET cache and shared mode's encoded
bindings cache, including their trace digests. Both structural fast-path caches
remain enabled. This separates structural reuse and delivery from differences
in exact-repeat cache policy, which matter when source sessions are replicated.
Report this ablation separately from the default-cache performance results.

For correctness checks, `--verify` converts every fast-path hit normally and
compares the result byte for byte. Do not enable it for performance timing.
PIM and EVENT traces retain their fallback converter paths.

Run modes sequentially on a comparable host. Repeat key comparisons and
alternate their order. Once runs have finished:

```bash
python analysis/session_scale.py compare \
  outputs/scale256-legacy-fast outputs/scale256-shared-fast
```

Comparison requires successful completion, identical configuration, dataset and
Python source hashes, and identical request CSVs, completed-call counts and
simulated clocks. It does not compare simulated clocks across different
workloads or cluster sizes.

## Show that the cluster is busy

`load.json` measures **replica batch occupancy**, including compute,
communication and memory waits. This is not SM utilization. The observer is
limited to colocated PP=1 instances without explicit DP groups.

- Busy fractions integrate completed batch intervals over the requested
  window, including idle time. Per-replica fractions show load imbalance.
- Per-bin mean active replicas comes from the same integrated intervals.
  Counts at bin boundaries give a sampled peak, not an exact unsampled peak.
- The companion queue CSV samples running and arrived-waiting requests at
  event times, with explicit timestamps. Sampling may be irregular when the
  simulator skips idle time. Queue means are sample means, not exact integrals.
- Live progress reports exclude still-running batch intervals and are labeled
  lower bounds. Use the final report for conclusions.
- A window ending after simulation completion is marked incomplete.

Report occupancy, queue behavior and batch sizes alongside wall time, CPU,
RSS, filesystem output and template statistics. A fixed light workload can
measure the cost of managing idle replicas but cannot establish loaded-cluster
scalability. A finite closed-loop workload also cannot establish an unbounded
open-loop overload threshold from a brief queue spike.
