#!/usr/bin/env bash
# Submit sim.py jobs to HTCondor.
# - Sweeps modes (default: 0,40) and indices [INDEX_START, INDEX_END]
#   with chunk size --per-task (default: 10)
# - Logs to ${HOME}/log
# - CLI: --modes "0,40"  --index-start N  --index-end N  --per-task N  --target STR
#        --shear F  --kappa F  --rot N  --layout STR  --band STR  --dry-run

set -euo pipefail

declare -a MODES=(0 40)     # override with: --modes "0,40"
INDEX_START=0               # override with: --index-start N
INDEX_END=2999              # inclusive; override with: --index-end N
PER_TASK=10                 # number of IDs per job; override with: --per-task N
TARGET="g1"                 # --target STR
SHEAR="0.02"                # --shear F
KAPPA="0.0"                 # --kappa F
ROT="0"                     # --rot N
LAYOUT="random"             # --layout STR
BAND="u,g,r,i,z,y"          # --band STR

PYTHON_EXE_PATH="$(command -v python3 || true)"
SCRIPT_PATH="sim.py"
LOG_DIR="${HOME}/log"
DRY_RUN=false

usage() {
  cat <<USAGE
Usage: $(basename "$0") [options]

Options:
  --index-start N       start index (default: ${INDEX_START})
  --index-end N         end index inclusive (default: ${INDEX_END})
  --script STD          python script to run (default: ${SCRIPT_PATH})
  --per-task N          number of IDs processed per job (default: ${PER_TASK})
  --layout STR          layout (default: ${LAYOUT})
  --target STR          target (default: ${TARGET})
  --shear F             shear (default: ${SHEAR})
  --kappa F             kappa (default: ${KAPPA})
  --rot N               rotation (default: ${ROT})
  --band STR            band (default: ${BAND})
  --dry-run             print the generated submit file and exit
  -h, --help            show this help
  --modes "M1,M2,..."   replace default modes (e.g. "0,40")

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
    --modes)        IFS=',' read -r -a MODES <<< "$2"; shift 2 ;;
    --index-start)  INDEX_START="$2"; shift 2 ;;
    --index-end)    INDEX_END="$2"; shift 2 ;;
    --per-task)     PER_TASK="$2"; shift 2 ;;
    --script)       SCRIPT_PATH="$2"; shift 2 ;;
    --target)       TARGET="$2"; shift 2 ;;
    --shear)        SHEAR="$2"; shift 2 ;;
    --kappa)        KAPPA="$2"; shift 2 ;;
    --rot)          ROT="$2"; shift 2 ;;
    --layout)       LAYOUT="$2"; shift 2 ;;
    --band)         BAND="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=true; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

# Resolve relative --script paths against the current working directory
# so HTCondor workers (which start in a different CWD) can find the file.
if [[ "$SCRIPT_PATH" != /* ]]; then
  SCRIPT_PATH="${PWD}/${SCRIPT_PATH}"
fi

# -------------------------------
# Sanity checks
# -------------------------------
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
request_memory  = 1228
request_cpus    = 1

executable      = /bin/bash
arguments       = "-c 'for ((i=\$(start); i<\$(end); i++)); do ${PYTHON_EXE_PATH} ${SCRIPT_PATH} --mode \$(mode) --rot ${ROT} --shear ${SHEAR} --kappa ${KAPPA} --target ${TARGET} --start \$i --end \$((i+1)) --layout ${LAYOUT} --band ${BAND} || exit 1; done'"
output          = ${LOG_DIR}/\$(ClusterId)_\$(ProcId)_\$(mode)_idx\$(start).out
error           = ${LOG_DIR}/\$(ClusterId)_\$(ProcId)_\$(mode)_idx\$(start).err
log             = ${LOG_DIR}/\$(ClusterId).log

queue mode, start, end from (
$(for m in "${MODES[@]}"; do
    for i in $(seq "${INDEX_START}" "${INDEX_END}"); do
      s=$((i*PER_TASK)); e=$((s+PER_TASK))
      printf "  %s %d %d\n" "$m" "$s" "$e"
    done
  done)
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
