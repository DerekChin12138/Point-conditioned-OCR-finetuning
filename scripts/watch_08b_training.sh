#!/usr/bin/env bash
# Watchdog for the 0.8B SFT pipeline (WSL2 + 15GB RAM + 8GB VRAM).
#
#   bash scripts/watch_08b_training.sh
#
# It watches the *supervisor* (scripts/supervise_08b.sh). If host memory goes
# critical it kills only the training child (the supervisor then auto-resumes
# from the latest checkpoint) — it never kills the supervisor, and it exits only
# when the supervisor is gone.
#
# Detached:
#   setsid nohup bash scripts/watch_08b_training.sh > logs/q08b_watchdog.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/logs/q08b_watchdog.log"
PIPELINE_LOG="${PIPELINE_LOG:-$ROOT/logs/q08b_pipeline.latest.log}"

MEM_FLOOR_MB="${MEM_FLOOR_MB:-800}"     # kill training if MemAvailable below this ...
MEM_FLOOR_HITS="${MEM_FLOOR_HITS:-3}"   # ... for N consecutive 30s samples
COOLDOWN_S="${COOLDOWN_S:-90}"          # pause after a kill so RAM can recover
GPU_WARN_MB="${GPU_WARN_MB:-7300}"
GPU_HARD_MB="${GPU_HARD_MB:-7900}"
SUPERVISOR_PAT="${WATCH_PAT:-${SUPERVISOR_PAT:-supervise_08b.sh}}"
KILL_PATS="${KILL_PATS:-run_08b_pipeline.sh unsloth_stage_a.py}"

low_hits=0
say() { echo "[$(date '+%F %T')] $*"; }

say "watchdog start  supervisor~$SUPERVISOR_PAT  mem_floor=${MEM_FLOOR_MB}MB x${MEM_FLOOR_HITS}  gpu_warn=${GPU_WARN_MB}MB"

kill_training() {
  say "killing target(s): $KILL_PATS"
  local p
  for p in $KILL_PATS; do pkill -TERM -f "$p" 2>/dev/null; done
  sleep 15
  for p in $KILL_PATS; do pkill -KILL -f "$p" 2>/dev/null; done
}

while true; do
  if ! pgrep -f "$SUPERVISOR_PAT" >/dev/null 2>&1; then
    say "supervisor gone — watchdog exit"
    exit 0
  fi

  avail=$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)
  read -r gpumem gpuutil < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr ',' ' ')
  step_line=$(tr '\r' '\n' < "$PIPELINE_LOG" 2>/dev/null | grep -E "[0-9]+/[0-9]+ \[" | tail -1)

  if tr '\r' '\n' < "$PIPELINE_LOG" 2>/dev/null | tail -40 | grep -qiE "Traceback|CUDA error|out of memory|device not ready|make_resident"; then
    say "ALERT: failure signature in pipeline log"
    tr '\r' '\n' < "$PIPELINE_LOG" | tail -12 >> "$OUT"
  fi

  if [[ "${gpumem:-0}" -ge "$GPU_HARD_MB" ]]; then
    say "ALERT: GPU ${gpumem}MiB >= ${GPU_HARD_MB}MiB (util ${gpuutil}%)"
  elif [[ "${gpumem:-0}" -ge "$GPU_WARN_MB" ]]; then
    say "warn: GPU ${gpumem}MiB (util ${gpuutil}%)"
  fi

  if [[ "${avail:-99999}" -lt "$MEM_FLOOR_MB" ]]; then
    low_hits=$((low_hits + 1))
    say "warn: MemAvailable ${avail}MB < ${MEM_FLOOR_MB}MB (hit ${low_hits}/${MEM_FLOOR_HITS})"
    if [[ "$low_hits" -ge "$MEM_FLOOR_HITS" ]]; then
      kill_training
      low_hits=0
      say "cooldown ${COOLDOWN_S}s"
      sleep "$COOLDOWN_S"
      continue
    fi
  else
    low_hits=0
  fi

  if [[ $(( $(date +%s) / 30 % 10 )) -eq 0 ]]; then
    say "hb mem_avail=${avail}MB gpu=${gpumem}MiB/${gpuutil}%  $step_line"
  fi
  sleep 30
done
