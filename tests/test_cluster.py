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

import anacal
import lsst.geom as geom
import numpy as np
import pytest
from lsst.skymap.ringsSkyMap import (
    RingsSkyMap,
    RingsSkyMapConfig,
)

from xlens.analysis.cluster import HaloMcBiasMultibandPipe as ClusterPipe
from xlens.simulator.catalog import (
    CatalogHaloTask,
    CatalogHaloTaskConfig,
)
from xlens.simulator.sim import (
    MultibandSimConfig,
    MultibandSimTask,
)

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


def _measure(exposure):
    gal_array = np.asarray(exposure.image.array, dtype=np.float64)
    lsst_psf = exposure.getPsf()
    center = geom.Point2D(PATCH_DIM // 2, PATCH_DIM // 2)
    psf_array = np.asarray(
        anacal.utils.resize_array(
            lsst_psf.computeImage(center).getArray(), (64, 64),
        ),
        dtype=np.float64,
    )
    fpfs_config = anacal.fpfs.FpfsConfig(
        sigma_shapelets=0.52,
        sigma_shapelets1=0.45,
        sigma_shapelets2=0.55,
    )
    return anacal.fpfs.process_image(
        fpfs_config=fpfs_config,
        mag_zero=MAG_ZERO,
        gal_array=gal_array,
        psf_array=psf_array,
        pixel_scale=PIXEL_SCALE,
        noise_variance=NOISE_VARIANCE,
        noise_array=None,
        detection=None,
    )


def _radial_shear(out, skymap):
    wcs = skymap[TRACT_ID].getWcs()
    ra, dec = wcs.pixelToSkyArray(out["x"], out["y"], degrees=True)
    ra_lens = wcs.getSkyOrigin().getRa().asDegrees()
    dec_lens = wcs.getSkyOrigin().getDec().asDegrees()

    angle = ClusterPipe.position_angle_ccw_from_east(
        ra_lens, dec_lens, ra, dec,
    ).rad
    dist = ClusterPipe.angsep(ra, dec, ra_lens, dec_lens)

    e1 = out["fpfs_e1"]
    e2 = out["fpfs_e2"]
    w = out["fpfs_w"]
    e1_g1 = out["fpfs_de1_dg1"]
    e2_g2 = out["fpfs_de2_dg2"]
    w_g1 = out["fpfs_dw_dg1"]
    w_g2 = out["fpfs_dw_dg2"]

    eT, eX = ClusterPipe._rotate_spin_2_vec(e1, e2, angle)
    r11, r22 = ClusterPipe._get_response_from_w_and_der(
        e1, e2, w, e1_g1, e2_g2, w_g1, w_g2,
    )
    rT, rX = ClusterPipe._rotate_spin_2_vec(r11, r22, angle)
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
    out = _measure(exposure)
    assert len(out) > 50, f"too few detections: {len(out)}"

    eT, eX, rT, rX, w, dist = _radial_shear(out, skymap)
    edges = np.linspace(5.0, 50.0, 5)
    gT, gX, counts = _bin_shear(eT, eX, rT, rX, w, dist, edges)

    assert np.all(counts > 0), f"empty radial bins: counts={counts}"
    assert np.all(np.isfinite(gT)), f"non-finite gT: {gT}"
    assert np.all(np.isfinite(gX)), f"non-finite gX: {gX}"
    assert gT[0] > 0.0, f"inner bin gT not positive: {gT}"
    assert gT[0] > gT[-1], f"gT not decreasing inner→outer: {gT}"


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
