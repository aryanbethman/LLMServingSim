# Session-scale reproducibility checklist — 2026-10-28

## Scope

Internal team handoff of the session generator, load observer, legacy graph-cache
ablation and experiment runner. This is not a blind submission or a badging claim.
The existing simulator build and its ASTRA-Sim/Chakra revisions are prerequisites.
Python requirements are captured in `analysis/requirements.txt`; each run also
records its environment, binary/profile hashes, source hashes and Git status.

## Static audit

`test-research-code/scripts/repro_check.py analysis --json` was run before and
after the handoff documentation was added. The initial scoped audit reported
README, ENV, DATA and RESULTS_CMD missing. The final audit reports README, ENV,
SEED, ENTRYPOINT, DATA and RESULTS_CMD all OK, with pinned Python dependencies.

The scoped audit does not see repository-level `tests/`. Test evidence below
comes from actual execution, not the static audit's `tests_present` flag.

## Executed correctness checks

| Check | Observed result |
| --- | --- |
| `python -m unittest discover -s tests -v` | All 115 tests passed |
| `PYTHON=env/bin/python ./serving/validate.sh` | All recorded legacy clocks match; all eight accuracy CSV/summary digests match |
| `EXTRA_ARGS="--legacy-value-fastpath" LLMSS_VERIFY_VALUE_FASTPATH=1 ./serving/validate.sh --clocks-only` | All recorded legacy clocks match |
| Supported non-DP scenarios with shared templates, overlay and both verification flags | All supported clocks match |
| Four-mode smoke on three complete source chains | All 60 request rows and 92,328,439,282 simulated ns match |
| Independent audit of the 800-session replicated input | All 16,000 calls and 200,786,272 token IDs verified against declared source/namespace mapping |
| Zero-to-positive memory-size regressions | Shared and legacy fast paths match fresh conversion across repeated zero/positive transitions |
| Timeout guard with an intentionally tiny time budget | Failed status recorded and only the invocation's scratch directory removed |
| Documentation production build and rendered-markup checker | Passed |

The full validation initially lacked matplotlib; it passed after the dependency
was installed and recorded. Generated example logs/plots are not source changes
and are excluded from the handoff. Verification-mode timings are correctness
evidence, not performance measurements.

## Data and fidelity

The calibration source is the existing Qwen SWE-bench corpus, already capped at
20 calls per chain. Replication does not truncate chains further. Its manifest
records selected source IDs, arrival seed/rate, SHA-256s and token policy.
The primary policy isolates repeated corpora while retaining all prefix
equalities within each corpus copy. This is a declared cache workload choice,
not a claim that repeated tasks are new independent recorded trajectories.

The repeatable timing cohort is a fixed seed-42 random subset of six complete
source chains, selected without using their timings. The 16-GPU case uses six
sessions at 0.2 sessions/s; the 256-GPU case uses sixteen corpus copies (96
sessions) at 3.2 sessions/s. This keeps source mix, expected arrival-window
duration and offered load per replica matched while bounding repeated-run cost.
The larger calibration corpus uses all source tasks and is reported separately.

## Measurement gates

Only completed runs with the expected call count, identical per-request CSVs
and clocks, and matching data/configuration/runtime hashes enter a comparison.
Use a complete measurement window and final integrated batch occupancy to call
a workload busy. Running-request snapshots and live lower bounds are provisional.
Batch occupancy is not physical GPU SM utilization. A finite closed-loop trace
does not by itself establish an open-loop saturation threshold.

All matched timing modes use the same controller protocol, profile, filesystem,
retention policy and load observer. Artifact counting and input cleanup happen
after the timed simulator process and are reported separately. Resource-aborted
calibration attempts are recorded and excluded from successful timing pairs.
The full 800-session pilot reached its declared one-hour host budget before
completion. Its live occupancy lower bound demonstrates offered load, but its
runtime is excluded from completed-run performance comparisons. The original
runner failed while parsing an empty GNU-time file after termination; this was
fixed and the timeout/cleanup path was checked directly. The pilot's separate
failure record preserves the original error and labels missing metrics.

See `docs/docs/workloads/session-scaling.md` and `analysis/README.md` for the
actual runnable entrypoints. Final performance conclusions belong in the results
report after the timed campaign completes; this checklist does not assert them.
