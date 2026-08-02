# This file is part of xlens.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""PSF HSM-moment measurement utilities (see the usage notes in
the first comment block below).

Split out of ``xlens.utils.image``, which re-exports every public name here
for backward compatibility.
"""

from dataclasses import dataclass, field
from typing import Any

import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom as lsst_geom
import lsst.meas.extensions.shapeHSM  # noqa: F401  (registers HSM plugins)
import numpy as np
from lsst.meas.algorithms import KernelPsf
from numpy.typing import NDArray



# ---------------------------------------------------------------------------
# PSF HSM-moment measurement utilities
# ---------------------------------------------------------------------------
#
# Thin wrappers around ``lsst.meas.extensions.shapeHSM``'s
# ``HsmPsfMomentsPlugin`` and ``HigherOrderMomentsPSFPlugin``, packaged so
# every xlens task that wants per-source / per-cell / per-exposure PSF
# moments can share the same plugin wiring, schema, and DRP-compatible
# column naming (``{band}_ext_shapeHSM_HsmPsfMoments_{xx,yy,xy,...}`` and
# ``{band}_ext_shapeHSM_HigherOrderMomentsPSF_{pq,flag}``).
#
# Usage from a PipelineTask:
#
#   # in config.setDefaults:
#   default_psf_hsm_plugin_config(self.psfHsmMeasurement)
#
#   # in task __init__ (after the subtask exists):
#   schema = afwTable.SourceTable.makeMinimalSchema()
#   self.makeSubtask("psfHsmMeasurement", schema=schema,
#                    algMetadata=dafBase.PropertyList())
#   self._psfHsmCtx = build_psf_hsm_context(
#       schema, self.config.psfHsmMeasurement)
#
#   # in the per-cell / per-exposure loop:
#   moments = measure_psf_hsm_moments(
#       self._psfHsmCtx, self.psfHsmMeasurement, exposure,
#   )
#   block = broadcast_psf_hsm_moments(moments, band, n=len(cat))
#   cat = np.asarray(rfn.merge_arrays([cat, block], flatten=True))


def _enumerate_higher_order_pq(min_order: int, max_order: int) -> list[tuple[int, int]]:
    """Same (p,q) enumeration as
    ``HigherOrderMomentsPlugin._get_pq_full`` — order 0..n, p from 0..n."""
    out = []
    for n in range(min_order, max_order + 1):
        for p in range(n + 1):
            out.append((p, n - p))
    return out


@dataclass
class PsfHsmContext:
    """Cached schema keys + (p, q) list for repeated PSF-HSM
    measurements. Build once per task instance via
    :func:`build_psf_hsm_context`."""
    schema: Any
    higher_order_pq: list[tuple[int, int]]
    key_ixx: Any
    key_iyy: Any
    key_ixy: Any
    key_flag: Any
    higher_order_keys: list[Any] = field(default_factory=list)
    higher_order_flag_key: Any = None


def default_psf_hsm_plugin_config(measurement_config) -> None:
    """Configure a ``SingleFrameMeasurementTask`` config block to run
    the DRP-equivalent PSF HSM plugin set on cell / exposure PSF stamps.

    The plugin list is
    ``base_PeakCentroid + ext_shapeHSM_HsmPsfMoments +
    ext_shapeHSM_HigherOrderMomentsPSF``; everything else is disabled
    so we don't try to measure fluxes / source shapes on the PSF stamp.
    ``HigherOrderMomentsPSF`` defaults to (min_order=3, max_order=4).
    To widen, set
    ``cfg.plugins['ext_shapeHSM_HigherOrderMomentsPSF'].max_order = N``.
    """
    measurement_config.plugins.names = [
        "base_PeakCentroid",
        "ext_shapeHSM_HsmPsfMoments",
        "ext_shapeHSM_HigherOrderMomentsPSF",
    ]
    measurement_config.slots.centroid = "base_PeakCentroid"
    measurement_config.slots.shape = None
    measurement_config.slots.psfShape = "ext_shapeHSM_HsmPsfMoments"
    measurement_config.slots.modelFlux = None
    measurement_config.slots.apFlux = None
    measurement_config.slots.calibFlux = None
    measurement_config.slots.gaussianFlux = None
    measurement_config.slots.psfFlux = None
    measurement_config.doReplaceWithNoise = False


def build_psf_hsm_context(schema, measurement_config) -> PsfHsmContext:
    """Gather the field keys + (p, q) list once after a
    ``SingleFrameMeasurementTask`` has registered its plugins on
    ``schema``."""
    ho_cfg = measurement_config.plugins["ext_shapeHSM_HigherOrderMomentsPSF"]
    pq = _enumerate_higher_order_pq(ho_cfg.min_order, ho_cfg.max_order)
    return PsfHsmContext(
        schema=schema,
        higher_order_pq=pq,
        key_ixx=schema["ext_shapeHSM_HsmPsfMoments_xx"].asKey(),
        key_iyy=schema["ext_shapeHSM_HsmPsfMoments_yy"].asKey(),
        key_ixy=schema["ext_shapeHSM_HsmPsfMoments_xy"].asKey(),
        key_flag=schema["ext_shapeHSM_HsmPsfMoments_flag"].asKey(),
        higher_order_keys=[
            schema[f"ext_shapeHSM_HigherOrderMomentsPSF_{p}{q}"].asKey()
            for (p, q) in pq
        ],
        higher_order_flag_key=schema[
            "ext_shapeHSM_HigherOrderMomentsPSF_flag"
        ].asKey(),
    )


def make_psf_stamp_exposure(psf_image) -> Any:
    """Wrap a centred PSF stamp (``lsst.afw.image.ImageD`` — typically
    ``cell.psf_image``) in a tiny ``ExposureF`` whose ``getPsf()``
    returns a ``KernelPsf`` over that exact stamp.

    Used to feed cell-coadd PSF stamps into the HSM measurement
    machinery. The synthetic exposure is ~few KB and is GC'd as soon
    as the caller releases it.
    """
    bbox = psf_image.getBBox()
    exp = afwImage.ExposureF(bbox)
    exp.image.array[:, :] = psf_image.array.astype(np.float32, copy=False)
    exp.variance.array[:, :] = 1.0
    exp.mask.array[:, :] = 0
    exp.setPsf(KernelPsf(afwMath.FixedKernel(psf_image)))
    return exp


def measure_psf_hsm_moments(
    ctx: PsfHsmContext,
    measurement_subtask,
    exposure,
    *,
    center: lsst_geom.Point2D | None = None,
) -> dict[str, float | bool]:
    """Run ``HsmPsfMoments + HigherOrderMomentsPSF`` once on
    ``exposure`` at ``center`` (default: ``exposure.getBBox().getCenter()``).

    The ``HsmPsfMomentsPlugin`` evaluates
    ``exposure.getPsf().computeKernelImage(center)``
    by default (``useSourceCentroidOffset=False``), so the PSF is sampled
    on the pixel grid with no subpixel shift — one exposure, one PSF.

    Returns a flat ``dict`` keyed by the raw plugin field names (no band
    prefix); call :func:`broadcast_psf_hsm_moments` to attach a band-
    prefixed copy to a source catalog.
    """
    if center is None:
        bbox = exposure.getBBox()
        center = lsst_geom.Point2D(bbox.getMinX() + bbox.getWidth() / 2.0,
                                   bbox.getMinY() + bbox.getHeight() / 2.0)
    cx = int(round(center.getX()))
    cy = int(round(center.getY()))

    cat = afwTable.SourceCatalog(ctx.schema)
    rec = cat.addNew()
    # 1x1 footprint around the seed pixel; PeakCentroid only reads the
    # brightest Peak's coordinates and HsmPsfMoments asks the PSF model
    # directly — neither walks the Footprint pixels, so we don't need
    # to span the full exposure (which on a coadd patch would be ~12k
    # rows of SpanSet).
    seed_box = lsst_geom.Box2I(lsst_geom.Point2I(cx, cy), lsst_geom.Extent2I(1, 1))
    footprint = afwDetection.Footprint(afwGeom.SpanSet(seed_box))
    psf_img = exposure.image
    bbox = exposure.getBBox()
    if bbox.contains(lsst_geom.Point2I(cx, cy)):
        peak_val = float(psf_img.array[cy - bbox.getMinY(),
                                       cx - bbox.getMinX()])
    else:
        peak_val = 1.0
    footprint.addPeak(float(cx), float(cy), peak_val)
    rec.setFootprint(footprint)

    try:
        measurement_subtask.run(cat, exposure)
    except Exception:
        # Plugin-internal failures already toggle the flag bit; harness
        # errors fall through and the flag-driven NaN replacement below
        # preserves the row shape.
        pass

    flag = bool(rec.get(ctx.key_flag))
    ho_flag = bool(rec.get(ctx.higher_order_flag_key))
    out: dict[str, float | bool] = {
        "ext_shapeHSM_HsmPsfMoments_xx": float(rec.get(ctx.key_ixx)),
        "ext_shapeHSM_HsmPsfMoments_yy": float(rec.get(ctx.key_iyy)),
        "ext_shapeHSM_HsmPsfMoments_xy": float(rec.get(ctx.key_ixy)),
        "ext_shapeHSM_HsmPsfMoments_flag": flag,
        "ext_shapeHSM_HigherOrderMomentsPSF_flag": ho_flag,
    }
    for (p, q), k in zip(ctx.higher_order_pq, ctx.higher_order_keys):
        out[f"ext_shapeHSM_HigherOrderMomentsPSF_{p}{q}"] = (
            np.nan if ho_flag else float(rec.get(k))
        )
    return out


def broadcast_psf_hsm_moments(
    moments: dict[str, float | bool],
    band: str,
    n: int,
    survey: str | None = None,
) -> NDArray:
    """Replicate one PSF-moment dict into an ``n``-row structured
    array with DRP-style column names ``{prefix}_<raw key>`` (``prefix`` =
    ``{survey}_{band}`` when ``survey`` is given, else ``{band}``), ready for
    ``rfn.merge_arrays`` into a per-band source catalog."""
    prefix = f"{survey}_{band}" if survey is not None else band
    dtype_fields: list[tuple[str, Any]] = []
    for k, v in moments.items():
        dt = np.bool_ if isinstance(v, (bool, np.bool_)) else np.float64
        dtype_fields.append((f"{prefix}_{k}", dt))
    out = np.empty(n, dtype=np.dtype(dtype_fields))
    for k, v in moments.items():
        out[f"{prefix}_{k}"] = v
    return out
