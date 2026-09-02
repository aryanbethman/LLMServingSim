#!/usr/bin/env bash
# Run one persistent fixed-workload simulator-scale experiment with monitoring.
set -u

if [[ $# -ne 4 ]]; then
  echo "usage: $0 RESULT_DIR CLUSTER_CONFIG LOGICAL_NPUS TEMPLATE_CACHE_MAX_ENTRIES" >&2
  exit 2
fi

result=$1
cluster_config=$2
logical_npus=$3
template_cache_max_entries=$4
repo=/home/marvell/LLMServingSim
dataset=dataset/sharegpt_req750_rate10_llama.jsonl

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

/usr/bin/time -v timeout --preserve-status 7200 \
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
