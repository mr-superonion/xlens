"""Edge margin for EXTERNAL detection catalogs in ``MeasureCoaddsPipe``.

Internal detection never returns a source within ``config.anacal.bound``
pixels of the image edge.  An external catalog must get the same margin:
rows that close to the edge are dropped before forced measurement (their
stamps would hang off the image), every other row is measured exactly as
before, and ``bound = 0`` keeps every row.

Also covers ``config.external_detection_id_column``: the ids of an
external catalog become the output ``object_id``; without it the ids are
generated as before.
"""

import anacal
import numpy as np
import numpy.lib.recfunctions as rfn
import pytest
from lsst.skymap.discreteSkyMap import DiscreteSkyMap, DiscreteSkyMapConfig

from xlens.processor.measure_coadds import (
    MeasureCoaddsPipe,
    MeasureCoaddsPipeConfig,
)
from xlens.simulator.catalog import CatalogShearTask, CatalogShearTaskConfig
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask
from xlens.utils.handle import make_exposure_handles

NPIX = 600
PIXEL_SCALE = 0.2
BOUND = 35


def _pipe(bound, id_column=None):
    config = MeasureCoaddsPipeConfig()
    config.external_detection_id_column = id_column
    config.anacal.force_size = True
    config.anacal.force_center = True
    config.anacal.do_noise_bias_correction = False
    config.anacal.bound = bound
    config.fpfs.do_noise_bias_correction = False
    config.fpfs.sigma_shapelets1 = 0.52
    return MeasureCoaddsPipe(config=config)


def test_partition_keep_box():
    """Rows outside keep_box are dropped; the rest keep their region and
    their input row index."""
    task = _pipe(BOUND)
    # two side-by-side regions tiling [0, 200) x [0, 100)
    regions = [(0, 0, 0, 100, 100), (1, 100, 0, 200, 100)]
    x = np.array([5.0, 34.9, 35.0, 99.0, 100.0, 164.9, 165.0, 50.0, 50.0])
    y = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 34.9, 65.0])
    det = np.zeros(len(x), dtype=[("x1_det", "f8"), ("x2_det", "f8")])
    det["x1_det"] = x * PIXEL_SCALE
    det["x2_det"] = y * PIXEL_SCALE
    keep_box = (BOUND, BOUND, 200 - BOUND, 100 - BOUND)

    cats, order = task._partition_external_detection(
        det, regions, PIXEL_SCALE, keep_box=keep_box,
    )
    # kept: x in [35, 165) and y in [35, 65)
    np.testing.assert_array_equal(order[0], [2, 3])
    np.testing.assert_array_equal(order[1], [4, 5])
    assert len(cats[0]) == 2 and len(cats[1]) == 2

    # without a box nothing but the tiling limits the rows
    cats, order = task._partition_external_detection(
        det, regions, PIXEL_SCALE,
    )
    np.testing.assert_array_equal(order[0], [0, 1, 2, 3, 7, 8])
    np.testing.assert_array_equal(order[1], [4, 5, 6])


def test_external_detection_edge_margin():
    config = DiscreteSkyMapConfig()
    config.projection = "TAN"
    config.raList = [0.0]
    config.decList = [0.0]
    config.radiusList = [0.49 * NPIX * PIXEL_SCALE / 3600.0]
    config.rotation = 0.0
    config.patchInnerDimensions = [NPIX, NPIX]
    config.patchBorder = 0
    config.pixelScale = PIXEL_SCALE
    config.tractOverlap = 0.0
    skymap = DiscreteSkyMap(config=config)
    assert tuple(skymap[0].getNumPatches()) == (1, 1)

    # a grid that reaches (and crosses) the image edge
    cconfig = CatalogShearTaskConfig()
    cconfig.layout = "grid"
    cconfig.extend_ratio = 1.08
    truth = CatalogShearTask(config=cconfig).run(
        tract_info=skymap[0], seed=0,
    ).truthCatalog

    sconfig = MultibandSimConfig()
    sconfig.survey_name = "lsst"
    sconfig.force_isotropic = True
    sconfig.draw_image_noise = False
    exposure = MultibandSimTask(config=sconfig).run(
        tract_info=skymap[0], patch_id=0, band="i", seed=0,
        truthCatalog=truth,
    ).simExposure

    detection = anacal.table.make_catalog_empty(
        np.zeros(len(truth)), np.zeros(len(truth))
    )
    detection["ra"] = truth["ra"]
    detection["dec"] = truth["dec"]
    detection["wsel"] = 1.0

    def run(bound, id_column=None, det=detection):
        return np.asarray(_pipe(bound, id_column).run(
            exposure_handles_dict=make_exposure_handles(
                exposure, tract=0, patch=0, band="i",
            ),
            corr_array=None,
            skyMap=skymap,
            tract=0,
            patch=0,
            detection=det,
        ).anacalCatalog)

    cat_all = run(0)
    cat_cut = run(BOUND)

    wcs = skymap[0].getWcs()

    def pixels(cat):
        return wcs.skyToPixelArray(cat["ra"], cat["dec"], degrees=True)

    def inside(cat):
        x, y = pixels(cat)
        return (
            (x >= BOUND) & (x < NPIX - BOUND)
            & (y >= BOUND) & (y < NPIX - BOUND)
        )

    # the test only means something if the margin holds sources
    keep = inside(cat_all)
    assert 0 < keep.sum() < len(cat_all)
    # bound = 0 keeps every detection that lies on the image
    x, y = wcs.skyToPixelArray(truth["ra"], truth["dec"], degrees=True)
    on_image = (x >= 0) & (x < NPIX) & (y >= 0) & (y < NPIX)
    assert len(cat_all) == on_image.sum()

    # with the margin: exactly the rows outside it, in the same order ...
    assert len(cat_cut) == keep.sum()
    assert inside(cat_cut).all()
    np.testing.assert_array_equal(cat_cut["ra"], cat_all["ra"][keep])
    np.testing.assert_array_equal(cat_cut["dec"], cat_all["dec"][keep])
    # ... and measured exactly as without it
    for name in ("lsst_i_fpfs1_e1", "lsst_i_fpfs1_e2",
                 "lsst_i_fpfs1_de1_dg1", "lsst_i_fpfs1_de2_dg2"):
        np.testing.assert_array_equal(cat_cut[name], cat_all[name][keep])

    # object ids.  Default: generated, consecutive within the patch.
    assert np.all(np.diff(cat_cut["object_id"]) == 1)

    # With an id column: the ids of the detections that were measured.
    ids = 7 * np.arange(len(truth), dtype=np.int64) + 3
    det_id = rfn.append_fields(detection, "truth_id", ids, usemask=False)
    cat_id = run(BOUND, id_column="truth_id", det=det_id)
    assert len(cat_id) == len(cat_cut)
    row = (cat_id["object_id"] - 3) // 7
    np.testing.assert_array_equal(cat_id["ra"], truth["ra"][row])
    np.testing.assert_array_equal(cat_id["dec"], truth["dec"][row])
    # nothing else changes
    for name in cat_cut.dtype.names:
        if name != "object_id":
            np.testing.assert_array_equal(cat_id[name], cat_cut[name])

    # the column must exist, be integer and be unique
    with pytest.raises(ValueError, match="no such column"):
        run(BOUND, id_column="truth_id")
    bad = rfn.append_fields(
        detection, "truth_id", np.zeros(len(truth), dtype=np.int64),
        usemask=False,
    )
    with pytest.raises(ValueError, match="repeated values"):
        run(BOUND, id_column="truth_id", det=bad)
    bad = rfn.append_fields(
        detection, "truth_id", np.arange(len(truth), dtype=np.float64),
        usemask=False,
    )
    with pytest.raises(ValueError, match="integer column"):
        run(BOUND, id_column="truth_id", det=bad)
