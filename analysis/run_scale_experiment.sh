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
repo=/home/marvell/LLMServingSim
dataset=dataset/sharegpt_req750_rate10_llama.jsonl

if [[ "$timeout_seconds" != none && ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
  echo "TIMEOUT_SECONDS must be a positive integer or none" >&2
  exit 2
fi

if [[ -e "$result" ]] && { [[ -e "$result/run.sh" ]] || [[ -e "$result/manifest.json" ]] || [[ -e "$result/exit_status" ]]; }; then
  echo "result directory already contains run artifacts: $result" >&2
  exit 2
fi
mkdir -p "$result"
cp "$0" "$result/run.sh"
cd "$repo"

/home/marvell/LLMServingSim/env/bin/python3 analysis/write_run_manifest.py \
  --repo "$repo" \
  --config "$cluster_config" \
  --dataset "$dataset" \
  --output "$result/manifest.json" \
  --logical-npus "$logical_npus" \
  --template-cache-max-entries "$template_cache_max_entries"

printf '%s\n' "timeout_seconds=$timeout_seconds" > "$result/runner_settings.txt"
timeout_prefix=()
if [[ "$timeout_seconds" != none ]]; then
  timeout_prefix=(timeout --preserve-status "$timeout_seconds")
fi

/usr/bin/time -v "${timeout_prefix[@]}" \
  env PATH=/home/marvell/LLMServingSim/env/bin:$PATH \
  /home/marvell/LLMServingSim/env/bin/python3 main.py \
  --cluster-config "$cluster_config" \
  --dataset "$dataset" \
  --num-req 750 \
  --network-backend analytical \
  --execution-template-mode shared-template \
  --template-bundle-builder fused \
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
