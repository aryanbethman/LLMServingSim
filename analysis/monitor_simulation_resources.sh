#!/usr/bin/env bash
# Persist host-side resource samples for one simulator process tree.
set -u

usage() {
  echo "usage: $0 --root-pid PID --result-dir DIR [--interval-seconds N]" >&2
  exit 2
}

root_pid=""
result_dir=""
interval_seconds=5
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-pid) root_pid=${2:-}; shift 2 ;;
    --result-dir) result_dir=${2:-}; shift 2 ;;
    --interval-seconds) interval_seconds=${2:-}; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$root_pid" =~ ^[0-9]+$ && -n "$result_dir" && "$interval_seconds" =~ ^[0-9]+$ && "$interval_seconds" -gt 0 ]] || usage

output="$result_dir/host_resources.csv"
started_epoch=$(date +%s)
printf '%s\n' 'epoch_s,elapsed_s,process_count,total_rss_kb,astra_rss_kb,python_rss_kb,total_vsz_kb,total_cpu_pct,open_fds,result_file_count,result_bytes,filesystem_free_kb,filesystem_free_inodes' > "$output"

descendants() {
  ps -eo pid=,ppid= | awk -v root="$root_pid" '
    { parent[$1] = $2 }
    END {
      for (pid in parent) {
        current = pid
        hops = 0
        while (current != root && (current in parent) && hops++ < 256) {
          current = parent[current]
        }
        if (current == root || pid == root) print pid
      }
    }'
}

while true; do
  mapfile -t pids < <(descendants)
  [[ ${#pids[@]} -gt 0 ]] || break

  total_rss=0
  astra_rss=0
  python_rss=0
  total_vsz=0
  total_cpu=0
  open_fds=0
  process_count=0
  for pid in "${pids[@]}"; do
    line=$(ps -p "$pid" -o rss=,vsz=,pcpu=,args= 2>/dev/null || true)
    [[ -n "$line" ]] || continue
    read -r rss vsz cpu command <<< "$line"
    [[ "$rss" =~ ^[0-9]+$ && "$vsz" =~ ^[0-9]+$ ]] || continue
    total_rss=$((total_rss + rss))
    total_vsz=$((total_vsz + vsz))
    total_cpu=$(awk -v total="$total_cpu" -v value="$cpu" 'BEGIN { printf "%.2f", total + value }')
    case "$command" in
      *AstraSim*) astra_rss=$((astra_rss + rss)) ;;
      *main.py*) python_rss=$((python_rss + rss)) ;;
    esac
    if [[ -d "/proc/$pid/fd" ]]; then
      fd_count=$(find "/proc/$pid/fd" -maxdepth 1 -type l 2>/dev/null | wc -l | tr -d ' ')
      open_fds=$((open_fds + fd_count))
    fi
    process_count=$((process_count + 1))
  done

  now_epoch=$(date +%s)
  elapsed=$((now_epoch - started_epoch))
  result_file_count=$(find "$result_dir" -type f 2>/dev/null | wc -l | tr -d ' ')
  result_bytes=$(du -sb "$result_dir" 2>/dev/null | awk '{print $1}')
  filesystem_free_kb=$(df -Pk "$result_dir" | awk 'NR == 2 {print $4}')
  filesystem_free_inodes=$(df -Pi "$result_dir" | awk 'NR == 2 {print $4}')
  printf '%s\n' "$now_epoch,$elapsed,$process_count,$total_rss,$astra_rss,$python_rss,$total_vsz,$total_cpu,$open_fds,$result_file_count,$result_bytes,$filesystem_free_kb,$filesystem_free_inodes" >> "$output"
  sleep "$interval_seconds"
done
