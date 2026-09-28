#!/usr/bin/env bash
# Self-healing supervisor for the 0.8B two-round SFT pipeline.
#
#   setsid nohup bash scripts/supervise_08b.sh > logs/q08b_supervise.log 2>&1 < /dev/null &
#
# Runs: q1 → merge-q1 → q2 → merge-q2 → eval-q1 → eval-q2.
# If a training stage dies (WSL glitch, watchdog memory kill, ...), it waits for
# RAM to recover and resumes from the latest checkpoint-* automatically.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export UNSLOTH_CE_LOSS_N_CHUNKS="${UNSLOTH_CE_LOSS_N_CHUNKS:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRIES="${MAX_TRIES:-10}"
MIN_AVAIL_MB="${MIN_AVAIL_MB:-2500}"

RUNDIR_Q1="${RUNDIR_Q1:-checkpoints/${RUN_Q1:-q1_08b}}"
RUNDIR_Q2="${RUNDIR_Q2:-checkpoints/${RUN_Q2:-q2_08b}}"
PIPE_LOG="logs/q08b_pipeline_$(date +%Y%m%d_%H%M%S).log"
ln -sf "$(basename "$PIPE_LOG")" logs/q08b_pipeline.latest.log
exec >> "$PIPE_LOG" 2>&1

say() { echo "[$(date '+%F %T')] [supervisor] $*"; }
avail() { awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo; }
latest_ckpt() { ls -d "$1"/checkpoint-* 2>/dev/null | sort -V | tail -1; }
is_done() { [[ -f "$1/adapter_final/adapter_config.json" ]]; }

wait_ram() {
  local a
  while :; do
    a=$(avail)
    [[ "$a" -ge "$MIN_AVAIL_MB" ]] && return 0
    say "waiting for RAM: MemAvailable=${a}MB < ${MIN_AVAIL_MB}MB"
    sleep 20
  done
}

run_stage() {  # <q1|q2> <rundir>
  local stage="$1" rundir="$2" tries=0 ck res rc
  while :; do
    if is_done "$rundir"; then say "$stage already complete ($rundir/adapter_final)"; return 0; fi
    wait_ram
    ck="$(latest_ckpt "$rundir")"
    say "$stage attempt $((tries + 1)) resume=${ck:-none}"
    if [[ "$stage" == "q1" ]]; then
      CONFIRM=1 NUM_WORKERS="$NUM_WORKERS" RESUME_Q1="$ck" bash train/run_08b_pipeline.sh q1
    else
      CONFIRM=1 NUM_WORKERS="$NUM_WORKERS" RESUME_Q2="$ck" bash train/run_08b_pipeline.sh q2
    fi
    rc=$?
    if is_done "$rundir"; then say "$stage DONE (rc=$rc)"; return 0; fi
    tries=$((tries + 1))
    if [[ "$tries" -ge "$MAX_TRIES" ]]; then say "$stage FAILED after $tries tries (rc=$rc)"; return 1; fi
    say "$stage exited rc=$rc without adapter_final — retrying in 30s"
    sleep 30
  done
}

say "=== supervise start (workers=$NUM_WORKERS, max_tries=$MAX_TRIES) ==="
say "pipeline log = $PIPE_LOG"

run_stage q1 "$RUNDIR_Q1" || { say "ABORT at q1"; exit 1; }

say "merge-q1"
bash train/run_08b_pipeline.sh merge-q1 || { say "ABORT at merge-q1"; exit 1; }

run_stage q2 "$RUNDIR_Q2" || { say "ABORT at q2"; exit 1; }

say "merge-q2"
bash train/run_08b_pipeline.sh merge-q2 || { say "ABORT at merge-q2"; exit 1; }

say "eval-q1"
bash train/run_08b_pipeline.sh eval-q1 || say "eval-q1 failed (non-fatal)"
say "eval-q2"
bash train/run_08b_pipeline.sh eval-q2 || say "eval-q2 failed (non-fatal)"

say "=== ALL DONE ==="
tail -5 "$PIPE_LOG" 2>/dev/null || true
