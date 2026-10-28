# Session scale experiments

Use the repository's simulator installation, including its pinned ASTRA-Sim
and Chakra submodules. The environment used for each run is captured in that
run's `environment.txt`; the observed Python dependency snapshot is also kept
in `requirements.txt` beside this README.

Dataset: generate source sessions using `python -m workloads.generators swebench`
or supply an existing session JSONL. The SWE-bench generator downloads published
trajectories from its documented source bucket; it requires the tokenizer
dependencies described in the workload documentation. For example:

```bash
python -m workloads.generators swebench \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --num-sessions 50 --sps 0.2 --max-calls 20 --seed 42 \
  --output outputs/source-sessions.jsonl
```

The source is already capped at the requested number of calls. Replication
preserves complete chains **as supplied**, without truncating them further.
Raw workloads and run outputs are not committed. Manifests record input and
output checksums and provenance; verify the source checksum for an exact replay.

Entrypoints:

```bash
python -m workloads.generators replicate-sessions --help
python analysis/session_scale.py select-source --help
python analysis/session_scale.py prepare --help
python analysis/session_scale.py run --help
python analysis/session_scale.py compare --help
python -m unittest discover -s tests -v
```

For exact reproduction commands, mode definitions, seeds, cache policy,
measurement definitions and limits, see
[`docs/docs/workloads/session-scaling.md`](../docs/docs/workloads/session-scaling.md).
The root README remains the minimal project front door.
