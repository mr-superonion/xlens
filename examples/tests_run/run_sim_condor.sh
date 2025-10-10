#!/usr/bin/env bash
# Submit sim.py jobs to HTCondor.
# - Sources setup.sh (so your env + PATH/LD_LIBRARY_PATH/etc propagate)
# - Sweeps mode in {0,40} and index in [0,49]
# - Writes logs to ./log/

set -euo pipefail

# -------------------------------
# Config (edit these as desired)
# -------------------------------
SCRIPT_PATH="${SCRIPT_PATH:-sim.py}"      # path to sim.py (relative or absolute)
SETUP_SH="${SETUP_SH:-setup.sh}"          # env setup file to source
MODES=(0 40)
INDEX_START=${INDEX_START:-0}
INDEX_END=${INDEX_END:-49}                # inclusive
TARGET="${TARGET:-g1}"
SHEAR="${SHEAR:-0.02}"
KAPPA="${KAPPA:-0.0}"
ROT="${ROT:-0}"
LAYOUT="${LAYOUT:-random}"

# -------------------------------
# Environment & sanity checks
# -------------------------------
if [[ -z "${PSCRATCH:-}" ]]; then
  echo "Error: PSCRATCH environment variable is not set." >&2
  exit 1
fi

if [[ ! -f "$SETUP_SH" ]]; then
  echo "Error: setup file '$SETUP_SH' not found." >&2
  exit 1
fi

# Load environment locally so condor can inherit it via getenv=true
# shellcheck source=/dev/null
source "$SETUP_SH"

PYTHON_EXE_PATH="$(command -v python3 || true)"
if [[ -z "$PYTHON_EXE_PATH" ]]; then
  echo "Error: python3 not found in PATH after sourcing '$SETUP_SH'." >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: sim script '$SCRIPT_PATH' not found." >&2
  exit 1
fi

# -------------------------------
# Logs
# -------------------------------
LOG_DIR="log"
mkdir -p "$LOG_DIR"
LOG_FILE_BASENAME="$(basename "$SCRIPT_PATH" .py)"

# -------------------------------
# Build submit description
# -------------------------------
# We’ll pass variables (mode, index, index_end) via HTCondor’s "queue from" list.
# Use late materialization macros: $(mode) etc inside 'arguments'.

condor_submit <<EOF
universe        = vanilla
initialdir      = ${PWD}
notification    = never
getenv          = true
request_memory  = 4096

# Use the python interpreter directly; pass the script + args in 'arguments'
executable      = ${PYTHON_EXE_PATH}

# Nice, descriptive logs per proc
output          = ${LOG_DIR}/\$(mode)_idx\$(index)_\$(ClusterId)_\$(ProcId).out
error           = ${LOG_DIR}/\$(mode)_idx\$(index)_\$(ClusterId)_\$(ProcId).err
log             = ${LOG_DIR}/\$(ClusterId).log

# Arguments to python:
#   python sim.py --mode $(mode) --start $(index) --end $(index_end) ...
arguments       = ${SCRIPT_PATH} \
  --mode \$(mode) \
  --rot ${ROT} \
  --shear ${SHEAR} \
  --kappa ${KAPPA} \
  --target ${TARGET} \
  --start \$(index) \
  --end \$(index_end) \
  --layout ${LAYOUT}

# Generate the Cartesian product of (mode, index) and provide index_end=index+1
# We materialize all triplets here.
queue mode, index, index_end from (
$(for m in "${MODES[@]}"; do
    for i in $(seq "${INDEX_START}" "${INDEX_END}"); do
      echo "  ${m} ${i} $((i+1))"
    done
  done)
)
EOF
