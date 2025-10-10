#!/usr/bin/env bash
# Submit sim.py jobs to HTCondor.
# - Sources setup on the worker node
# - Sweeps mode in {0,40} and index in [0,5]
# - Logs to ./log/

set -euo pipefail

# -------------------------------
# Config
# -------------------------------
MODES=(0 40)
INDEX_START=${INDEX_START:-0}
INDEX_END=${INDEX_END:-5}   # inclusive
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

SCRIPT_PATH="sim.py"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: sim script '$SCRIPT_PATH' not found." >&2
  exit 1
fi

# -------------------------------
# Logs
# -------------------------------
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# -------------------------------
# Submit description
# -------------------------------
condor_submit <<EOF
universe        = vanilla
initialdir      = ${PWD}
notification    = never
getenv          = true
request_memory  = 4096

# Run a login shell
executable      = /bin/bash

# Single line arguments. Condor will expand \$(mode) etc on the worker.
arguments       = python3 ${SCRIPT_PATH} --mode \$(mode) --rot ${ROT} --shear ${SHEAR} --kappa ${KAPPA} --target ${TARGET} --start \$(index) --end \$(index_end) --layout ${LAYOUT}

output          = ${LOG_DIR}/\$(mode)_idx\$(index)_\$(ClusterId)_\$(ProcId).out
error           = ${LOG_DIR}/\$(mode)_idx\$(index)_\$(ClusterId)_\$(ProcId).err
log             = ${LOG_DIR}/\$(ClusterId).log

# Cartesian product of (mode, index) with index_end = index+1
queue mode, index, index_end from (
$(for m in "${MODES[@]}"; do
    for i in $(seq "${INDEX_START}" "${INDEX_END}"); do
      echo "  ${m} ${i} $((i+1))"
    done
  done)
)
EOF
