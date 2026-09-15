# Fork migration: what moved, what's broken

> Historical assessment from 2026-09-13. Repairs on 2026-09-14 supersede
> the broken-runner and missing-guard findings below. File-free DP is now
> explicitly rejected rather than implemented. Converted profile attention
> now uses complete self-contained source surfaces and legacy timing keys;
> the earlier coarse export changed off-grid and long-prefill timings.
> Captured wire logs also identified asynchronous stdout diagnostics corrupting
> compact READY records; these diagnostics now use stderr and the five failed
> compact-protocol clock scenarios pass after rebuilding both analytical backends.
> See MIGRATION.md for current behavior. Submodule publication and general
> concurrent-run safety remain outstanding. The original findings
> below are retained as an audit record, not current operating instructions.

The Marvell/HiPC fork (`~/LLMServingSim`, branch `feature/tiered-memory-topology`,
built on upstream 89f312f) was moved onto upstream a4053bc, 129 commits later.
Upstream renamed most of the tree in between: `main.py` became `python -m serving`,
`inference_serving/` became `serving/core/`, `cluster_config/` became
`configs/cluster/`, and `dataset/` became `workloads/`.

`MIGRATION.md` lists what was ported and why some things were dropped. This file
covers what I checked on 2026-09-13 and what is still wrong. Where the two
disagree, this file is the one I tested.

## State of the tree

Nothing is committed. Every change in the main repo is either a modified file or
an untracked one (`serving/core/execution_templates.py`,
`serving/core/tiered_memory.py`, `tests/`, the H100 profiles, `analysis/`, the
new cluster configs).

The submodule pointers are also at risk. `astra-sim` is at `a69b559` and its
nested `chakra` at `d57c55e`. Neither commit is on any remote branch, and the
only remote configured in this tree is upstream `casys-kaist`. In the old tree
the ASTRA commits were pushed to `aryan/feature/tiered-memory-topology`; the
rebased ones haven't been. A fresh clone can't check these commits out until
they're pushed somewhere.

## The four headline flags

| Flag | Status |
|---|---|
| `--execution-template-mode shared-template` | Ported. Broken with DP groups (see below). |
| `--template-bundle-builder fused` | **Removed.** Passing it is a startup error. |
| `--compact-controller-protocol` | Ported, same wire format. |
| `--template-cache-max-entries 128` | Ported, same validation and release handling. |

Shared-template mode now always uses the fused builder
(`convert_rows_to_template_bundle` in `serving/core/graph_generator.py`). The
old `payload` builder still exists as `build_template_bundle()` but only the
unit tests call it. So the fused speedup is always on and you can drop the flag
from commands. Anything that still passes it fails with
`unrecognized arguments: --template-bundle-builder fused`.

I checked that the fused path builds the same thing the payload builder would.
For every batch in two real runs, I built the bundle both ways and compared
them. All 718 bundles matched (276 on `single_node_tp_pp_instance`, 442 on
`single_node_4_instance_2TP`), and both runs hit their recorded clocks exactly.
Nothing in `tests/` checks this, although the old fork had a component test
for it.

## What was verified

- `./env/bin/python -m unittest discover -s tests`: 25 tests pass.
- Clock suite with the full flag set (`shared-template`, compact protocol, cache
  128). I stopped it after 32 of 58 scenarios once the DP hang showed up. 30 of
  those 32 matched their baselines exactly. The two failures were `multi` and
  `saturated_wide_batch`, both the intermittent hang described below. Run on
  its own, `saturated_wide_batch` passes in both legacy and full-flags mode
  with the baseline clock (20350033685).
- MoE and the remaining scenarios after `tp_pp` in the suite were not run with
  the full flags, apart from the targeted DP runs below.

## Issues

### 1. DP groups hang in file-free modes

Any cluster config with a `dp_group` hangs under `--execution-template-mode
in-memory` or `shared-template`. Legacy mode is fine.

| Config | legacy | in-memory | shared-template |
|---|---|---|---|
| `single_node_dp_instance` | 1445267327 (matches) | hang | hang |
| `single_node_moe_dp_ep_instance` | 1479062704 (matches) | hang | hang |
| `single_node_moe_dp_tp_pp_instance` | 1642858815 (matches) | hang | hang |

Legacy finishes in about 4 s. The file-free runs were killed at 90-150 s,
reproducibly. The compact protocol and cache limit aren't the cause, because
plain `shared-template` hangs too.

The fork never had DP groups (`dp_group` appears nowhere in the old
`main.py` or `inference_serving/`). The DP branches for file-free mode in
`serving/__main__.py` (the `graph_payload` handoff around lines 950-965 and
1119-1125) were written during the migration and never worked. Legacy DP
writes every member's `.et` into one shared workload folder, so ASTRA sees one
workload. File-free mode sends each member's graph separately, and the round
never completes. I haven't root-caused it past that.

None of the fork's own configs (`baseline_scale_*`, `llama_405b_h100_tp8_pp2`,
`tiered_memory_pd_*`) use `dp_group`, so the fork's experiments aren't blocked.
The claim that validation with `EXTRA_ARGS="--execution-template-mode
shared-template"` is an equivalence check is wrong, though. The suite will
hang on the first DP scenario (`dp`).

While in there: the DP branch at `serving/__main__.py:952` has a nested
`if args.execution_template_mode == "legacy":` inside the same check, so its
`else` can never run. It's harmless but should be cleaned up alongside the fix.

### 2. `analysis/run_scale_experiment.sh` doesn't run

It passes `--template-bundle-builder fused` (a startup error, above) and uses
`dataset=dataset/sharegpt_req750_rate10_llama.jsonl`, but there's no `dataset/`
directory in this tree. Fix both before starting a scale run. `--num-req 750`
still works because argparse accepts it as a prefix of `--num-reqs`.

### 3. Intermittent hang under concurrent load

Now and then a run stops making progress with both processes blocked reading
from each other (`wchan=pipe_read` on the frontend and on ASTRA). I saw it on
`multi` and `saturated_wide_batch` while other simulator runs were going in the
same checkout, including once in plain legacy mode with none of the template
flags. It is not tied to the flags, `PYTHONHASHSEED`, or `--run-id`: re-running
the same seed and run-id passed every time, and 12 concurrent runs
(6 migrated, 6 pristine upstream) all passed.

I can't say whether this is new with the migration. Until it's understood, run
long experiments one at a time and use a timeout. Every stdin write in
`serving/core/controller.py` flushes, so it isn't an unflushed pipe.

### 4. Old commands need rewriting

Upstream renamed or removed flags, so the fork's old command lines won't parse:

| Old | New |
|---|---|
| `python main.py` | `python -m serving` |
| `--fp 16` | `--dtype bfloat16` |
| `--num-req` | `--num-reqs` (old spelling still accepted as a prefix) |
| `--max-batch` | `--max-num-seqs` |
| `--profile-variant nominal\|low\|high` | `--profile-variant bf16\|bf16-low\|bf16-high` (a folder name now) |
| `--template-bundle-builder` | removed, always fused |
| `--cleanup-consumed-traces`, `--retain-traces` | removed. Per-run inputs are cleaned by default; `--keep-inputs` keeps them |
| `--gen`, `--prioritize-prefill`, `--enable-attn-prediction` | removed upstream, no drop-in replacement |

The old `HANDOFF.md` 405B command uses four of these and won't work as written.
The working form is in `MIGRATION.md` under "Running it".

### 5. No backend guard for file-free modes

The fork refused `in-memory`/`shared-template` unless `--network-backend
analytical`. The new code has no such check, so `--network-backend ns3` with
shared-template will start and then fail in some less obvious way.

### 6. `MIGRATION.md` says both caches are active. In shared-template mode, only one is.

Upstream's converted-graph cache (`_ET_CACHE`) is only used in legacy mode.
`generate_graph` returns before the cache lookup for any other mode. In
shared-template mode the only deduplication is the template-id dedup.
That's probably fine for performance, but the doc overstates it.

### 7. The converted profiles are partly synthetic

These carry over from `MIGRATION.md` and still apply to every H100 result:

- `sampler` is 0 in converted bundles, which is about 25 us per iteration too
  optimistic.
- `skew_fit` is disabled, so heterogeneous-KV decode batches get no skew
  correction.
- v0's attention grid stops at kv = 2048. Anything longer is linear
  extrapolation.
- H100 tp8 is projected from tp1/tp2/tp4 (3.89% median, 21.78% P90 back-test
  error). 405B is a calibrated projection, not a measurement.

### 8. Environment quirks

Use `./env/bin/python` and put `env/bin` first on `PATH`. The machine-wide
editable `chakra` install points at the old fork's checkout and shadows this
tree's copy. Also, Chakra is installed as a package, so edits to
`llm_converter.py` do nothing until you reinstall it.

## Suggested order

1. Push the `astra-sim` and `chakra` submodule commits somewhere, then commit
   this tree.
2. Fix `run_scale_experiment.sh` (drop the builder flag, fix the dataset path).
3. Fix or explicitly reject DP groups in file-free modes. A startup error is
   better than a silent hang.
4. Add the ns3 guard back, and add a fused-vs-payload equivalence test.
5. Re-run the full clock suite with the full flags once DP is handled.
