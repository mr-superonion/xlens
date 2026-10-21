#!/usr/bin/env bash
# Submit summary_seeds.py jobs to HTCondor.
# - Sweeps seed indices [INDEX_START, INDEX_END] (each job gets
#   --min-id/--max-id) with chunk size --per-task (default: 10)
# - Logs to ${HOME}/log
# - Passes: --emax --layout --target --shear --width-max --z-bounds \
#   --pixel-scale --stamp-dim --bands --redshift
#   [--no-correction] --min-id --max-id

set -euo pipefail

INDEX_START=0               # override with: --index-start N
INDEX_END=2999              # inclusive; override with: --index-end N
PER_TASK=10                 # number of IDs per job; override with: --per-task N
TARGET="g1"                 # --target STR
SHEAR="0.02"                # --shear F
EMAX="0.30"                 # --emax F
LAYOUT="random"             # --layout STR
WIDTH_MAX="2.75"            # --width-max F
Z_BOUNDS="0.3,0.6,0.9,1.2,1.5,1.8"   # --z-bounds STR
PIXEL_SCALE="0.2"           # --pixel-scale F
STAMP_DIM="1350"            # --stamp-dim N (usable image dim in pixels)
BANDS="ugrizy"              # --bands STR
REDSHIFT="flexzboost"       # --redshift STR
VERSION=""                  # --version N (optional; empty = unversioned outputs)
TAG=""                      # --tag STR (optional; empty = unnamed outputs)
SUMMARY_TAG=""              # --summary-tag STR (optional; names the CUTS)
MAG_MAX=""                  # --mag-max F (optional; empty = script default)
TRACE_MIN=""                # --trace-min F (optional; empty = script default)
SHAPE_NAME=""               # --shape-name STR (optional; empty = summary.py default "fpfs")

PYTHON_EXE_PATH="$(command -v python3 || true)"
SCRIPT_PATH="summary.py"
LOG_DIR="${HOME}/log"
DRY_RUN=false
NO_CORR=false               # if true, append --no-correction to arguments

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
  --emax F              emax (default: ${EMAX})
  --width-max F         width maximum (default: ${WIDTH_MAX})
  --z-bounds STR        comma-separated redshift bounds (default: ${Z_BOUNDS})
  --pixel-scale F       pixel scale (arcsec/pixel, default: ${PIXEL_SCALE})
  --stamp-dim N         usable image dimension in pixels, for the area/density
                        calc (default: ${STAMP_DIM})
  --bands STR           bands used for photo-z estimation (default: ${BANDS})
  --redshift STR        photo-z estimator name (default: ${REDSHIFT})
  --version N           optional integer tag; injects --version N into the
                        python arguments (default: none)
  --tag STR             configuration name; injects --tag STR into the python
                        arguments, reading process_mode<N>_<tag>/ and writing
                        partials to summary-flux-40-00_<tag>/ (default: none)
  --summary-tag STR     name for the CUTS; injects --summary-tag STR, which
                        appends to the PARTIALS directory only, so several cut
                        choices over the same catalogs coexist (default: none)
  --mag-max F           injects --mag-max F (default: none = script default)
  --trace-min F         injects --trace-min F (default: none = script default)
  --shape-name STR      shape columns the estimator reads; injects
                        --shape-name STR (default: none = summary.py's
                        "fpfs"; e.g. "lsst_i_fpfs1")
  --no-correction       add --no-correction to summary_seeds.py arguments
  --dry-run             print the generated submit file and exit
  -h, --help            show this help

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
    --index-start)   INDEX_START="$2"; shift 2 ;;
    --index-end)     INDEX_END="$2"; shift 2 ;;
    --script)        SCRIPT_PATH="$2"; shift 2 ;;
    --per-task)      PER_TASK="$2"; shift 2 ;;
    --emax)          EMAX="$2"; shift 2 ;;
    --layout)        LAYOUT="$2"; shift 2 ;;
    --target)        TARGET="$2"; shift 2 ;;
    --shear)         SHEAR="$2"; shift 2 ;;
    --width-max)     WIDTH_MAX="$2"; shift 2 ;;
    --z-bounds)      Z_BOUNDS="$2"; shift 2 ;;
    --pixel-scale)   PIXEL_SCALE="$2"; shift 2 ;;
    --stamp-dim)     STAMP_DIM="$2"; shift 2 ;;
    --bands)         BANDS="$2"; shift 2 ;;
    --redshift)      REDSHIFT="$2"; shift 2 ;;
    --version)       VERSION="$2"; shift 2 ;;
    --tag)           TAG="$2"; shift 2 ;;
    --summary-tag)   SUMMARY_TAG="$2"; shift 2 ;;
    --mag-max)       MAG_MAX="$2"; shift 2 ;;
    --trace-min)     TRACE_MIN="$2"; shift 2 ;;
    --shape-name)    SHAPE_NAME="$2"; shift 2 ;;
    --no-correction) NO_CORR=true; shift ;;
    --dry-run)       DRY_RUN=true; shift ;;
    -h|--help)       usage; exit 0 ;;
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

# Optional arg for --no-correction
NO_CORR_ARG=""
if $NO_CORR; then
  NO_CORR_ARG="--no-correction"
fi

# -------------------------------
# Generate submit description
# -------------------------------
generate_submit() {
  cat <<EOF
universe        = vanilla
initialdir      = ${PWD}
notification    = never
getenv          = true
request_memory  = 1024
request_cpus    = 1
environment = OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;BLIS_NUM_THREADS=1;VECLIB_MAXIMUM_THREADS=1;TBB_NUM_THREADS=1;NUMEXPR_MAX_THREADS=1;NUMEXPR_NUM_THREADS=1;MKL_DYNAMIC=FALSE

executable      = ${PYTHON_EXE_PATH}
arguments       = ${SCRIPT_PATH} \\
                  --emax ${EMAX} \\
                  --layout ${LAYOUT} \\
                  --target ${TARGET} \\
                  --shear ${SHEAR} \\
                  --width-max ${WIDTH_MAX} \\
                  --z-bounds ${Z_BOUNDS} \\
                  --pixel-scale ${PIXEL_SCALE} \\
                  --stamp-dim ${STAMP_DIM} \\
                  --bands ${BANDS} \\
                  --redshift ${REDSHIFT} \\
                  ${VERSION:+--version ${VERSION}} \\
                  ${TAG:+--tag ${TAG}} \\
                  ${SUMMARY_TAG:+--summary-tag ${SUMMARY_TAG}} \\
                  ${MAG_MAX:+--mag-max ${MAG_MAX}} \\
                  ${TRACE_MIN:+--trace-min ${TRACE_MIN}} \\
                  ${SHAPE_NAME:+--shape-name ${SHAPE_NAME}} \\
                  ${NO_CORR_ARG} \\
                  --min-id \$(start) --max-id \$(end)
output          = ${LOG_DIR}/\$(ClusterId)_\$(ProcId)_seed\$(start).out
error           = ${LOG_DIR}/\$(ClusterId)_\$(ProcId)_seed\$(start).err
log             = ${LOG_DIR}/\$(ClusterId).log

queue start, end from (
$(
  for i in $(seq "${INDEX_START}" "${INDEX_END}"); do
    s=$((i*PER_TASK)); e=$((s+PER_TASK))
    printf "  %d %d\n" "$s" "$e"
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
