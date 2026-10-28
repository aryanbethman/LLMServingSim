# Session-scale cleanup decisions — 2026-10-28

The user approved the scoped implementation and cleanup plan. Existing simulator
behavior, routing, memory tiers, model profiles, dtype and scheduler defaults are
preserved. New graph-cache and load-observer behavior is opt-in. No merge or pull
request is authorized.

## Applied safe changes

- Added an analysis README, pinned observed Python requirements, an experiment
  configuration and a production-built documentation page. These address the
  static entrypoint/data/results/environment findings.
- Kept generated workloads, run output and temporary artifacts outside tracked
  source. AppleDouble files introduced by transfer were removed; normal runtime
  `__pycache__` directories remain ignored and do not enter a Git source export.
- Preserved the original uncommitted implementation and paper draft as separate
  commits before beginning the approved work; pinned both nested repositories.

## Audit findings reviewed without changing numerical behavior

The static release audit was scoped to `analysis/`, avoiding changing simulator
scratch directories. It is a heuristic audit, not an execution proof.

| Finding | Decision |
| --- | --- |
| `/tmp` lock filename flagged as a machine home path | Generic Linux temporary lock, not a user-specific path; retained |
| Existing comparator uses sets | Membership/difference checks and diagnostics, not simulation iteration order; left unchanged |
| Runner reads environment | Critical fast-path/cache settings are explicitly set and recorded; environment capture is intentional |
| Profile paths collected in a set | Sorted before manifest serialization; retained |
| Wall clock used | Metadata, monitoring and timeout accounting only; it is not a workload RNG seed |
| Ignored Python bytecode | Excluded from source exports; no deletion of active caches required |
| Generic config/layout suggestions | The CLI, explicit base configuration and README supply these interfaces; no broad repository reorganization |

No unrelated dead-code removal, RNG changes, thread changes, precision changes
or behavior-changing refactors were applied as cleanup. The explicit experiment
parameters are captured in manifests. Anonymization and archival deposition are
outside this internal handoff.
