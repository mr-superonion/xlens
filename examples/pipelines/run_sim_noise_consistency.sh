#!/usr/bin/env bash
# End-to-end check that  simulation(draw_image_noise=False) -> AddNoisePipe -> MeasureCoaddsPipe
# equals                 simulation(draw_image_noise=True)                  -> MeasureCoaddsPipe
# when run through the butler with pipetask (examples/pipelines/sim_noise_consistency.yaml).
#
# Builds a throw-away butler repo, registers LsstComCam (for the band records)
# and a small discrete skymap, runs both chains on tract 0 / patch 0 in r,i,z,
# and compares the exposures and the anacal catalogs bit for bit.
#
#   run_sim_noise_consistency.sh [REPO_DIR]        (default: ./sim_noise_consistency_repo)
#
# Needs the stackvana environment (butler, pipetask, xlens installed) and
# CATSIM_DIR pointing at the catsim input catalogs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${1:-$PWD/sim_noise_consistency_repo}"
PIPE="$HERE/sim_noise_consistency.yaml"
SKYMAP_CFG="$HERE/skymap_sim_test.py"
QUERY="skymap='sim_test' AND tract=0 AND patch=0 AND band IN ('r','i','z')"
: "${CATSIM_DIR:?CATSIM_DIR must point at the catsim input catalogs}"

echo "== repo: $REPO"
rm -rf "$REPO"
butler create "$REPO"
butler register-instrument "$REPO" lsst.obs.lsst.LsstComCam
butler register-skymap "$REPO" -C "$SKYMAP_CFG"

for chain in chainA chainB; do
  echo "== pipetask run: $chain"
  pipetask run -b "$REPO" -i skymaps -o "u/sim_noise/$chain" \
    -p "$PIPE#$chain" -d "$QUERY" -j 1 --register-dataset-types
done

echo "== compare"
python3 - "$REPO" <<'EOF'
import sys
import numpy as np
from lsst.daf.butler import Butler

repo = sys.argv[1]
bA = Butler(repo, collections=["u/sim_noise/chainA"])
bB = Butler(repo, collections=["u/sim_noise/chainB"])
did = dict(skymap="sim_test", tract=0, patch=0)
ok = True

for band in ("r", "i", "z"):
    a = bA.get("simA_0_rot0_coadd", **did, band=band)
    q = bB.get("simB_0_rot0_coadd", **did, band=band)
    b = bB.get("simB_0_rot0_coadd_noise0_coadd", **did, band=band)
    same_img = np.array_equal(a.image.array, b.image.array)
    same_var = np.array_equal(a.variance.array, b.variance.array)
    same_msk = np.array_equal(a.mask.array, b.mask.array)
    quiet_var = np.all(q.variance.array == a.variance.array)
    print(f"band {band}: noisy == noiseless+addNoise: image {same_img} variance {same_var} mask {same_msk}"
          f" | noiseless variance plane already correct: {quiet_var}"
          f" | max|diff| {np.abs(a.image.array - b.image.array).max():.3g}")
    ok &= same_img and same_var and same_msk

ca = bA.get("simA_meas_anacal_catalog", **did)
cb = bB.get("simB_meas_anacal_catalog", **did)
ca, cb = ca.as_array() if hasattr(ca, "as_array") else np.asarray(ca), cb.as_array() if hasattr(cb, "as_array") else np.asarray(cb)
same_cols = ca.dtype.names == cb.dtype.names
bad = [] if (same_cols and len(ca) == len(cb)) else ["<shape or columns differ>"]
if not bad:
    bad = [c for c in ca.dtype.names if not np.array_equal(ca[c], cb[c])]
print(f"anacal catalogs: {len(ca)} vs {len(cb)} rows; bit-identical: {not bad}" + (f"; differing: {bad[:6]}" if bad else ""))
ok &= not bad
print("RESULT:", "CONSISTENT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
EOF
