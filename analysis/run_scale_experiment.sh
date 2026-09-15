#!/usr/bin/env bash
# Run one persistent fixed-workload simulator-scale experiment with monitoring.
set -u

if [[ $# -ne 4 && $# -ne 5 ]]; then
  echo "usage: $0 RESULT_DIR CLUSTER_CONFIG LOGICAL_NPUS TEMPLATE_CACHE_MAX_ENTRIES [TIMEOUT_SECONDS|none]" >&2
  exit 2
fi

result=$1
cluster_config=$2
logical_npus=$3
template_cache_max_entries=$4
timeout_seconds=${5:-7200}
# Resolve the repo from this script's own location, so the experiment
# runs wherever the tree is checked out.
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dataset="$repo/experiments/hipc_upstream_control/sharegpt_req750_rate10_llama.jsonl"
monitor="$repo/analysis/monitor_simulation_resources.sh"
python_bin="${PYTHON:-$repo/env/bin/python}"
if [[ "$python_bin" != */* ]]; then
  python_bin=$(command -v "$python_bin") || {
    echo "PYTHON command not found" >&2
    exit 2
  }
fi
[[ "$result" == /* ]] || result="$PWD/$result"
[[ "$cluster_config" == /* ]] || cluster_config="$repo/$cluster_config"

if [[ "$timeout_seconds" != none && ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "TIMEOUT_SECONDS must be a positive integer or none" >&2
  exit 2
fi

if [[ -e "$result" ]] && { [[ -e "$result/run.sh" ]] || [[ -e "$result/manifest.json" ]] || [[ -e "$result/exit_status" ]]; }; then
  echo "result directory already contains run artifacts: $result" >&2
  exit 2
fi

for prerequisite in "$python_bin" "$cluster_config" "$dataset" "$monitor" analysis/write_run_manifest.py; do
  if [[ "$prerequisite" != /* ]]; then
    prerequisite="$repo/$prerequisite"
  fi
  if [[ ! -r "$prerequisite" ]]; then
    echo "missing or unreadable prerequisite: $prerequisite" >&2
    exit 2
  fi
done
if [[ ! -x "$python_bin" ]]; then
  echo "PYTHON is not executable: $python_bin" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "missing required command: timeout" >&2
  exit 2
fi
if [[ ! -x /usr/bin/time ]]; then
  echo "missing required command: /usr/bin/time" >&2
  exit 2
fi
mkdir -p "$result"
cp "$0" "$result/run.sh"
cd "$repo"

export PATH="$repo/env/bin:$PATH"
# Upstream's frontend prepends ../ after entering astra-sim, so its CLI
# needs repo-relative paths even though manifests use absolute paths.
cluster_arg=$("$python_bin" -c 'import os,sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' "$cluster_config" "$repo")
dataset_arg=experiments/hipc_upstream_control/sharegpt_req750_rate10_llama.jsonl

# PROFILE_VARIANT selects an uncertainty arm (e.g. bf16-low); unset keeps the
# dtype-derived variant, bf16 for these models.
profile_variant=${PROFILE_VARIANT:-}
variant_args=()
if [[ -n "$profile_variant" ]]; then
  variant_args=(--profile-variant "$profile_variant")
fi

"$python_bin" analysis/write_run_manifest.py \
  --repo "$repo" \
  --config "$cluster_config" \
  --dataset "$dataset" \
  --output "$result/manifest.json" \
  --logical-npus "$logical_npus" \
  --template-cache-max-entries "$template_cache_max_entries" \
  --profile-variant "${profile_variant:-bf16}"
manifest_status=$?
if [[ "$manifest_status" -ne 0 ]]; then
  echo "manifest creation failed; simulator was not started" >&2
  exit "$manifest_status"
fi

printf '%s\n' "timeout_seconds=$timeout_seconds" "profile_variant=${profile_variant:-bf16}" > "$result/runner_settings.txt"
timeout_prefix=()
if [[ "$timeout_seconds" != none ]]; then
  timeout_prefix=(timeout --preserve-status "$timeout_seconds")
fi

/usr/bin/time -v "${timeout_prefix[@]}" \
  "$python_bin" -m serving \
  --cluster-config "$cluster_arg" \
  --dataset "$dataset_arg" \
  --num-reqs 750 \
  "${variant_args[@]}" \
  --network-backend analytical \
  --execution-template-mode shared-template \
  --compact-controller-protocol \
  --template-cache-max-entries "$template_cache_max_entries" \
  --execution-template-stats-output "$result/template_transport.json" \
  --output "$result/requests.csv" \
  --log-level WARNING \
  > "$result/stdout.log" 2> "$result/stderr_time.log" &
simulation_pid=$!

"$repo/analysis/monitor_simulation_resources.sh" \
  --root-pid "$simulation_pid" \
  --result-dir "$result" \
  --interval-seconds 5 &
monitor_pid=$!

wait "$simulation_pid"
status=$?
wait "$monitor_pid" || true
printf '%s\n' "$status" > "$result/exit_status"
exit "$status"
