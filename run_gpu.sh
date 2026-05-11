#!/usr/bin/env bash
# run_gpu_job.sh — Launch Python jobs on a remote GPU server using a
# conda env and code stored on a shared NFS mount.
#
# Usage:
#   ./run_gpu_job.sh [options] -- <python_args...>
#
# Examples:
#   ./run_gpu_job.sh -s train.py -- --epochs 10
#   ./run_gpu_job.sh -m detach -s train.py -- --config configs/exp1.yaml
#   ./run_gpu_job.sh -m live   -s train.py
#   ./run_gpu_job.sh -m ssh    -s train.py    # interactive shell + run

set -euo pipefail

# ---------- defaults (edit these for your setup) ----------
# GPU_HOST="172.27.21.97"             # SSH alias or user@host
GPU_HOST="172.31.100.241"             # SSH alias or user@host
NAS_ROOT="/DATAX/divyaksh"               # same path on desktop & GPU server
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/DATA3/divyaksh/miniconda3/envs/open_unlearning}"   # full path to env (prefix-style)
CODE_DIR="${CODE_DIR:-$NAS_ROOT/Projects/unlearning/unlearning-calibration}"
LOG_DIR="${LOG_DIR:-$CODE_DIR/tty_logs}"                         # resolved after arg parsing; defaults to $CODE_DIR/logs
CONDA_SH="${CONDA_SH:-/DATA3/divyaksh/miniconda3/etc/profile.d/conda.sh}"

MODE="ssh"            # ssh | detach | live
SCRIPT=""             # script to run (relative to CODE_DIR or absolute)
INTERPRETER="python"  # python | bash — how to invoke SCRIPT
SSH_USER="${SSH_USER:-}"  # SSH username; if set, login as $SSH_USER@$GPU_HOST
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-30000}"  # consider GPU "free" if used <= this

# ---------- arg parsing ----------
usage() {
  cat <<EOF
Usage: $0 [-h HOST] [-u USER] [-e ENV_PATH] [-c CODE_DIR] [-l LOG_DIR] [-m MODE] [-i INTERPRETER] -s SCRIPT [-- script_args...]

Options:
  -h HOST         SSH host/alias for GPU server          (default: $GPU_HOST)
  -u USER         SSH username (login as USER@HOST)      (default: ${SSH_USER:-current user / SSH config})
  -e ENV_PATH     Full path to conda env on NAS          (default: $CONDA_ENV_PATH)
  -c CODE_DIR     Working directory on NAS               (default: $CODE_DIR)
  -l LOG_DIR      Directory for stdout/stderr logs       (default: \$CODE_DIR/logs)
  -m MODE         ssh | detach | live                    (default: $MODE)
                    ssh    = run synchronously over SSH (Ctrl-C kills job)
                    detach = run in background via nohup, return immediately
                    live   = run detached, then tail -f the log
  -i INTERPRETER  python | bash                          (default: $INTERPRETER)
  -s SCRIPT       Script to execute (required)
  --              Everything after -- is forwarded to the script as args
  -?              Show this help

Examples:
  $0 -s train.py -- --epochs 10 --lr 1e-4
  $0 -u divyaksh -s train.py -- --epochs 10
  $0 -m detach -s eval.py -- --ckpt runs/exp1/best.pt
  $0 -m live -s train.py -- --config configs/big.yaml
  $0 -i bash -s scripts/run_sweep.sh -- exp1 exp2 exp3
EOF
  exit 0
}

PY_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h) GPU_HOST="$2"; shift 2 ;;
    -u) SSH_USER="$2"; shift 2 ;;
    -e) CONDA_ENV_PATH="$2"; shift 2 ;;
    -c) CODE_DIR="$2"; shift 2 ;;
    -l) LOG_DIR="$2"; shift 2 ;;
    -m) MODE="$2"; shift 2 ;;
    -i) INTERPRETER="$2"; shift 2 ;;
    -s) SCRIPT="$2"; shift 2 ;;
    --) shift; PY_ARGS=("$@"); break ;;
    -\?|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

[[ -z "$SCRIPT" ]] && { echo "Error: -s SCRIPT is required" >&2; exit 2; }
[[ "$MODE" =~ ^(ssh|detach|live)$ ]] || { echo "Error: -m must be ssh|detach|live" >&2; exit 2; }
[[ "$INTERPRETER" =~ ^(python|bash)$ ]] || { echo "Error: -i must be python|bash" >&2; exit 2; }

# LOG_DIR follows CODE_DIR unless explicitly set via -l or env var
LOG_DIR="${LOG_DIR:-$CODE_DIR/logs}"

# Build SSH target: "user@host" if SSH_USER set, else just "host"
# (so SSH config / current username can still take over when -u is omitted)
if [[ -n "$SSH_USER" ]]; then
  SSH_TARGET="${SSH_USER}@${GPU_HOST}"
else
  SSH_TARGET="$GPU_HOST"
fi

# ---------- build job metadata ----------
JOB_TS="$(date +%Y%m%d_%H%M%S)"
SCRIPT_BASE="$(basename "$SCRIPT")"
SCRIPT_BASE="${SCRIPT_BASE%.*}"   # strip any extension (.py, .sh, etc.)
JOB_ID="${SCRIPT_BASE}_${JOB_TS}_$$"
LOG_FILE="$LOG_DIR/${JOB_ID}.log"
PID_FILE="$LOG_DIR/${JOB_ID}.pid"

# Quote each script arg safely for remote shell
printf -v SCRIPT_ARGS_QUOTED '%q ' "${PY_ARGS[@]:-}"

# ---------- remote command ----------
# Note: $LOG_FILE etc. are NAS paths visible identically on both sides.
read -r -d '' REMOTE_CMD <<EOF || true
set -euo pipefail

mkdir -p "$LOG_DIR"

# --- pick least-used GPU by free memory ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found on \$(hostname)" >&2
  exit 3
fi

GPU_ID=\$(nvidia-smi --query-gpu=index,memory.used \\
  --format=csv,noheader,nounits \\
  | sort -t, -k2 -n \\
  | head -n1 \\
  | awk -F, '{gsub(/ /,"",\$1); print \$1}')

if [[ -z "\$GPU_ID" ]]; then
  echo "ERROR: could not determine a free GPU" >&2
  exit 4
fi

echo "[\$(date '+%F %T')] host=\$(hostname) job=$JOB_ID gpu=\$GPU_ID" | tee -a "$LOG_FILE"

# --- activate conda env from NAS ---
# Use prefix activation so we don't depend on env being registered in ~/.condarc
source "$CONDA_SH"
conda activate "$CONDA_ENV_PATH"

cd "$CODE_DIR"
export CUDA_VISIBLE_DEVICES="\$GPU_ID"
export PYTHONUNBUFFERED=1   # ensure logs flush in real time

# --- run ---
$INTERPRETER "$SCRIPT" $SCRIPT_ARGS_QUOTED
EOF

# ---------- launch ----------
echo "Job:    $JOB_ID"
echo "Host:   $SSH_TARGET"
echo "Code:   $CODE_DIR"
echo "Env:    $CONDA_ENV_PATH"
echo "Run:    $INTERPRETER $SCRIPT"
echo "Log:    $LOG_FILE"
echo "Mode:   $MODE"
echo

case "$MODE" in
  ssh)
    # Foreground: stream output, Ctrl-C propagates to remote (-tt forces tty).
    ssh -tt "$SSH_TARGET" "bash -lc $(printf '%q' "$REMOTE_CMD")" \
      2>&1 | tee -a "$LOG_FILE"
    ;;

  detach)
    # Detach via nohup on remote; capture PID for status/kill later.
    DETACH_WRAPPER="nohup bash -lc $(printf '%q' "$REMOTE_CMD") \
      >>'$LOG_FILE' 2>&1 & echo \$! > '$PID_FILE'; disown"
    ssh "$SSH_TARGET" "$DETACH_WRAPPER"
    sleep 1
    REMOTE_PID="$(cat "$PID_FILE" 2>/dev/null || echo '?')"
    echo "Detached. Remote PID: $REMOTE_PID"
    echo "Tail log with:  tail -f '$LOG_FILE'"
    echo "Kill with:      ssh $SSH_TARGET 'kill $REMOTE_PID'"
    ;;

  live)
    # Detach + tail. Ctrl-C here stops the tail, NOT the job.
    DETACH_WRAPPER="nohup bash -lc $(printf '%q' "$REMOTE_CMD") \
      >>'$LOG_FILE' 2>&1 & echo \$! > '$PID_FILE'; disown"
    ssh "$SSH_TARGET" "$DETACH_WRAPPER"
    sleep 1
    REMOTE_PID="$(cat "$PID_FILE" 2>/dev/null || echo '?')"
    echo "Running detached as remote PID $REMOTE_PID. Tailing log (Ctrl-C to stop tail; job keeps running)..."
    echo "----"
    tail -F "$LOG_FILE"
    ;;
esac