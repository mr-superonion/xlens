#!/usr/bin/env bash
# Submit summary.py jobs to HTCondor.
# - Sweeps group indices [INDEX_START, INDEX_END] (each job gets
#   --group-start/--group-end) with chunk size --per-task (default: 10)
# - Logs to ${HOME}/log
# - Only passes: --emax --layout --target --shear --start --end

set -euo pipefail

INDEX_START=0               # override with: --index-start N
INDEX_END=2999              # inclusive; override with: --index-end N
PER_TASK=10                 # number of group_ids per job; override with: --per-task N
EMAX="0.30"                 # --emax F
LAYOUT="random"             # --layout STR
TARGET="g1"                 # --target STR
SHEAR="0.02"                # --shear F

PYTHON_EXE_PATH="$(command -v python3 || true)"
SCRIPT_PATH="summary.py"    # if you need to change it: --script path/to/summary.py
LOG_DIR="${HOME}/log"
DRY_RUN=false

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --index-start N     start index (default: ${INDEX_START})
  --index-end N       end index inclusive (default: ${INDEX_END})
  --script PATH       python script to run (default: ${SCRIPT_PATH})
  --per-task N        number of group_ids processed per job (default: ${PER_TASK})
  --emax F            emax (default: ${EMAX})
  --layout STR        layout (default: ${LAYOUT})
  --target STR        target (default: ${TARGET})
  --shear F           shear (default: ${SHEAR})
  --dry-run           print the generated submit file and exit
  -h, --help          show this help

Notes:
- python = python3 in PATH
- logs   = \$HOME/log
USAGE
}

# -------------------------------
# Parse CLI
# -------------------------------
if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage; exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --index-start)  INDEX_START="$2"; shift 2 ;;
    --index-end)    INDEX_END="$2"; shift 2 ;;
    --script)       SCRIPT_PATH="$2"; shift 2 ;;
    --per-task)     PER_TASK="$2"; shift 2 ;;
    --emax)         EMAX="$2"; shift 2 ;;
    --layout)       LAYOUT="$2"; shift 2 ;;
    --target)       TARGET="$2"; shift 2 ;;
    --shear)        SHEAR="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=true; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# -------------------------------
# Sanity checks
# -------------------------------
if [[ -z "${PSCRATCH:-}" ]]; then
  echo "Error: PSCRATCH environment variable is not set." >&2
  exit 1
fi
if [[ -z "$PYTHON_EXE_PATH" ]]; then
  echo "Error: python3 not found in PATH." >&2
  exit 1
fi
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: script '$SCRIPT_PATH' not found." >&2
  exit 1
fi
if (( INDEX_END < INDEX_START )); then
  echo "Error: --index-end (${INDEX_END}) < --index-start (${INDEX_START})." >&2
  exit 1
fi
if (( PER_TASK <= 0 )); then
  echo "Error: --per-task must be a positive integer (got ${PER_TASK})." >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

# -------------------------------
# Generate submit description
# -------------------------------
generate_submit() {
  cat <<EOF
universe        = vanilla
initialdir      = ${PWD}
notification    = never
getenv          = true
request_memory  = 10000
request_cpus    = 1

executable      = ${PYTHON_EXE_PATH}
arguments       = ${SCRIPT_PATH} --emax ${EMAX} --layout ${LAYOUT} --target ${TARGET} --shear ${SHEAR} --group-start \$(start) --group-end \$(end)
output          = ${LOG_DIR}/\$(ClusterId)_\$(ProcId)_grp\$(start).out
error           = ${LOG_DIR}/\$(ClusterId)_\$(ProcId)_grp\$(start).err
log             = ${LOG_DIR}/\$(ClusterId).log

queue start, end from (
$(
  s=${INDEX_START}
  while (( s <= INDEX_END )); do
    e=$((s + PER_TASK))
    if (( e > INDEX_END + 1 )); then
      e=$((INDEX_END + 1))
    fi
    printf "  %d %d\n" "$s" "$e"
    s=$((s + PER_TASK))
  done
)
)
EOF
}

# -------------------------------
# Submit or dry-run
# -------------------------------
if $DRY_RUN; then
  generate_submit
else
  generate_submit | condor_submit
fi
