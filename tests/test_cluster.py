"""Smoke test mirroring ``examples/cluster/example1_cluster_lensing.ipynb``
but with isotropic (intrinsically round) galaxies so that the only shape
signal is the lensing shear from the NFW halo.

Simulates a single-band patch of a tract centered on an NFW halo, runs FPFS
shape measurement, then uses the static helpers on
``xlens.analysis.cluster.HaloMcBiasMultibandPipe`` to compute tangential and
cross shears in radial bins. The intrinsic shape noise is zero, so a clean
positive ``gT`` signal should appear in the inner bins and decay outward,
while ``gX`` should remain consistent with zero.
"""

import os

import numpy as np
import pytest
from lsst.skymap.ringsSkyMap import (
    RingsSkyMap,
    RingsSkyMapConfig,
)

from xlens.analysis.cluster import HaloMcBiasMultibandPipe as ClusterPipe
from xlens.processor.measure_coadds import (
    MeasureCoaddsPipe,
    MeasureCoaddsPipeConfig,
)
from xlens.simulator.catalog import (
    CatalogHaloTask,
    CatalogHaloTaskConfig,
)
from xlens.simulator.sim import (
    MultibandSimConfig,
    MultibandSimTask,
)
from xlens.utils.handle import make_exposure_handles

PATCH_DIM = 1001
PIXEL_SCALE = 0.2
MAG_ZERO = 30.0
NOISE_VARIANCE = 0.37
TRACT_ID = 40000
PATCH_ID = 0
SEED = 120
HALO_MASS = 4e14
HALO_CONC = 3.8
Z_LENS = 0.25
Z_SOURCE = 1.0


def _build_skymap() -> RingsSkyMap:
    cfg = RingsSkyMapConfig()
    cfg.patchInnerDimensions = [PATCH_DIM, PATCH_DIM]
    cfg.tractOverlap = 0.0
    cfg.patchBorder = 0
    cfg.numRings = 5000
    cfg.pixelScale = PIXEL_SCALE
    cfg.projection = "TAN"
    return RingsSkyMap(config=cfg)


def _make_truth(skymap):
    cfg = CatalogHaloTaskConfig()
    cfg.mass = HALO_MASS
    cfg.conc = HALO_CONC
    cfg.z_lens = Z_LENS
    cfg.z_source = Z_SOURCE
    cfg.layout = "random"
    return CatalogHaloTask(config=cfg).run(
        tract_info=skymap[TRACT_ID], seed=SEED,
    ).truthCatalog


def _simulate(skymap, truth):
    cfg = MultibandSimConfig()
    cfg.survey_name = "lsst"
    cfg.force_isotropic = True
    return MultibandSimTask(config=cfg).run(
        tract_info=skymap[TRACT_ID],
        patch_id=PATCH_ID,
        band="i",
        seed=SEED,
        truthCatalog=truth,
    ).simExposure


def _measure(exposure, skymap):
    """Run the production measurement pipeline, exactly like the cluster
    example notebook: AnaCal detection (detector.h) + forced FPFS, with
    the detector's differentiable selection weight ``wsel``.
    """
    config = MeasureCoaddsPipeConfig()
    config.anacal.sigma_arcsec = 0.52   # detection / e1, e2, w kernel
    config.fpfs.sigma_shapelets1 = 0.45  # kernel 1
    config.fpfs.sigma_shapelets2 = 0.55  # kernel 2
    pipe = MeasureCoaddsPipe(config=config)
    handles = make_exposure_handles(
        exposure,
        tract=TRACT_ID,
        patch=PATCH_ID,
        band="i",
        skyMap=skymap,
    )
    return pipe.run(
        exposure_handles_dict=handles,
        corr_array=None,
        skyMap=skymap,
        tract=TRACT_ID,
        patch=PATCH_ID,
    ).anacalCatalog


def _radial_shear(out, skymap):
    wcs = skymap[TRACT_ID].getWcs()
    ra = np.asarray(out["ra"])
    dec = np.asarray(out["dec"])
    ra_lens = wcs.getSkyOrigin().getRa().asDegrees()
    dec_lens = wcs.getSkyOrigin().getDec().asDegrees()

    angle = ClusterPipe.position_angle_ccw_from_east(
        ra_lens, dec_lens, ra, dec,
    ).rad
    dist = ClusterPipe.angsep(ra, dec, ra_lens, dec_lens)

    e1 = out["fpfs_e1"]
    e2 = out["fpfs_e2"]
    w = out["wsel"]
    e1_g1 = out["fpfs_de1_dg1"]
    e2_g2 = out["fpfs_de2_dg2"]
    w_g1 = out["dwsel_dg1"]
    w_g2 = out["dwsel_dg2"]

    eT, eX = ClusterPipe._rotate_spin_2_vec(e1, e2, angle)
    r11, r22 = ClusterPipe._get_response_from_w_and_der(
        e1, e2, w, e1_g1, e2_g2, w_g1, w_g2,
    )
    # Responses rotate as a (diagonal) matrix, not as a spin-2 vector --
    # this matches what ClusterPipe.run itself does.
    rT, rX = ClusterPipe._rotate_spin_2_matrix(r11, r22, angle)
    return eT, eX, rT, rX, w, dist


def _bin_shear(eT, eX, rT, rX, w, dist, edges):
    gT = np.full(len(edges) - 1, np.nan)
    gX = np.full(len(edges) - 1, np.nan)
    counts = np.zeros(len(edges) - 1, dtype=int)
    for i in range(len(edges) - 1):
        mask = (dist >= edges[i]) & (dist < edges[i + 1])
        counts[i] = int(np.sum(mask))
        if counts[i] == 0:
            continue
        rT_sum = np.sum(rT[mask])
        rX_sum = np.sum(rX[mask])
        if rT_sum != 0.0:
            gT[i] = np.sum(w[mask] * eT[mask]) / rT_sum
        if rX_sum != 0.0:
            gX[i] = np.sum(w[mask] * eX[mask]) / rX_sum
    return gT, gX, counts


@pytest.mark.skipif(
    "CATSIM_DIR" not in os.environ,
    reason="CATSIM_DIR not set; the LSST sim needs the CatSim2017 catalog.",
)
def test_cluster_lensing_isotropic():
    skymap = _build_skymap()
    truth = _make_truth(skymap)
    exposure = _simulate(skymap, truth)
    out = _measure(exposure, skymap)
    assert len(out) > 50, f"too few detections: {len(out)}"

    eT, eX, rT, rX, w, dist = _radial_shear(out, skymap)
    # Start outside the strong-lensing core (< 15 arcsec): the linear
    # shear estimator saturates there and the innermost annulus holds
    # only a handful of sources.
    edges = np.linspace(15.0, 60.0, 4)
    gT, gX, counts = _bin_shear(eT, eX, rT, rX, w, dist, edges)

    assert np.all(counts > 0), f"empty radial bins: counts={counts}"
    assert np.all(np.isfinite(gT)), f"non-finite gT: {gT}"
    assert np.all(np.isfinite(gX)), f"non-finite gX: {gX}"
    assert gT[0] > 0.0, f"inner bin gT not positive: {gT}"
    assert gT[0] > gT[-1], f"gT not decreasing inner→outer: {gT}"


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
