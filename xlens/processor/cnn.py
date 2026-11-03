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

"""Forced shape measurement with the D4-equivariant CNN of wl_D4T_CNN
(``Forward8_fixW_CNN``), calibrated by AnaCal.

:class:`CnnMeasurementTask` measures the detections of one cell: the
cell is deconvolved by its PSF and re-smoothed to the Gaussian the
network was trained with (``anacal.image.ImageQ``), with the noise-bias
correction of the measurement pipe (the rotated noise copy, or the
spectral renoising with the patch noise correlation), a stamp is cut at
every detection, and the fused C++ runner ``anacal.math.D4ShapeNet``
returns (e1, e2) with their shear responses.  The model is the ``.npz``
export of ``wl_D4T_CNN/cpu_src`` (weights plus metadata: sigmas of the
weight maps, shared head, suppression taper, stamp size, training PSF),
so no PyTorch is needed at measurement time.
"""

import json
import threading
from typing import Any

import anacal
import numpy as np
from lsst.pex.config import Config, Field, FieldValidationError
from lsst.pipe.base import Task
from numpy.typing import NDArray

from ..utils.random import num_rot


def gaussian_weight_map(
    size: int,
    sigma: float,
    *,
    suppress: bool = True,
    r_cut: float = 24.0,
    suppress_order: int = 8,
) -> NDArray:
    """The weight maps of ``Forward8_fixW_CNN`` (``gaussian_weight_2d`` of
    wl_D4T_CNN): a centred Gaussian density of width ``sigma`` pixels
    (not normalised to unit sum), optionally multiplied by the smooth
    outer taper exp(-(r^2 / r_cut^2)^(order / 2)), symmetrised under the
    two reflections so that the centre pixel is the maximum."""
    y = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    yy, xx = np.meshgrid(y, y, indexing="ij")
    r2 = xx**2 + yy**2
    g = np.exp(-0.5 * r2 / sigma**2) / (2.0 * np.pi * sigma**2)
    if suppress:
        g = g * np.exp(-((r2 / r_cut**2) ** (suppress_order / 2)))
    return (g + g[::-1, ::-1] + g[::-1, :] + g[:, ::-1]) / 4.0


def load_cnn_export(path: str) -> dict:
    """Read a ``wl_D4T_CNN/cpu_src`` model export (``.npz``: float arrays
    plus a JSON ``metadata`` entry) into the pieces ``anacal.math.
    D4ShapeNet`` takes.

    Returns a dict with ``blocks`` (list of (conv_w, conv_b, norm_w,
    norm_b, skip_w | None)), ``head_e1`` / ``head_e2`` (four linear
    weights each; the shared head is used for both), ``wnorm`` /
    ``wpool`` (the weight maps), ``res``, ``stamp_size``, ``sigma_arcsec``
    (the training re-smoothing scale) and the raw ``metadata``.
    """
    with np.load(path, allow_pickle=False) as z:
        if "metadata" not in z.files:
            raise ValueError(f"{path}: not a cpu_src model export (no metadata)")
        meta = json.loads(bytes(z["metadata"]).decode("utf-8"))
        arrays = {k: np.asarray(z[k], dtype=np.float64) for k in z.files if k != "metadata"}
    arch = meta.get("architecture")
    if arch != "Forward8_fixW_CNN":
        raise ValueError(f"{path}: architecture {arch!r} is not supported (Forward8_fixW_CNN only)")
    mk = meta["model"]
    nblk = int(mk["num_layers"])
    blocks = []
    for i in range(nblk):
        p = f"blocks.{i}."
        blocks.append(
            (
                arrays[p + "conv.weight"],
                arrays[p + "conv.bias"],
                arrays[p + "norm.weight"].reshape(-1),
                arrays[p + "norm.bias"].reshape(-1),
                arrays.get(p + "skip.weight"),
            )
        )
    if mk.get("shared_head", False):
        head = [arrays[f"head_shared.net.{j}.weight"] for j in (1, 3, 5, 7)]
        heads = (head, head)
    else:
        heads = tuple([arrays[f"head_{key}.net.{j}.weight"] for j in (1, 3, 5, 7)] for key in ("e1", "e2"))
    n = int(meta["stamp_size"])
    taper = dict(
        suppress=bool(mk.get("suppress", True)),
        r_cut=float(mk.get("r_cut", 24.0)),
        suppress_order=int(mk.get("suppress_order", 8)),
    )
    return dict(
        blocks=blocks,
        head_e1=heads[0],
        head_e2=heads[1],
        wnorm=gaussian_weight_map(n, float(mk["nl_sigma"]), **taper),
        wpool=gaussian_weight_map(n, float(mk["gw_sigma"]), **taper),
        res=float(mk.get("res_factor", 0.1)),
        stamp_size=n,
        sigma_arcsec=float(meta.get("preprocessing", {}).get("sigma_arcsec", -1.0)),
        metadata=meta,
    )


class CnnMeasurementConfig(Config):
    """Configuration for :class:`CnnMeasurementTask`."""

    model_path = Field[str](
        doc=(
            "Path of the wl_D4T_CNN/cpu_src model export (.npz with "
            "weights and metadata). REQUIRED whenever the cnn subtask runs."
        ),
        default="",
    )
    sigma_arcsec = Field[float](
        doc=(
            "Width [arcsec] of the Gaussian the stamps are re-smoothed to "
            "before the network sees them. Negative: the training value "
            "recorded in the export (preprocessing.sigma_arcsec). The "
            "network is a pixel-domain model: on a pixel scale different "
            "from the training one, matching the kernel in PIXELS "
            "(training sigma_arcsec / training pixel_scale x this pixel "
            "scale) is closer to the training distribution than keeping "
            "the arcsec value."
        ),
        default=-1.0,
    )
    precision = Field[str](
        doc="float32 (the trained precision) or float64",
        default="float32",
    )
    tangents = Field[int](
        doc=(
            "2: shear responses only (the stamp is measured about the "
            "detection pixel); 4: also the position responses."
        ),
        default=2,
    )
    noiseId = Field[int](doc="Noise realization id", default=0)
    rotId = Field[int](doc="rotation id", default=0)

    def validate(self):
        super().validate()
        if self.precision not in ("float32", "float64"):
            raise FieldValidationError(
                self.__class__.precision, self, "precision must be float32 or float64"
            )
        if self.tangents not in (2, 4):
            raise FieldValidationError(self.__class__.tangents, self, "tangents must be 2 or 4")
        if self.rotId >= num_rot:
            raise FieldValidationError(
                self.__class__.rotId,
                self,
                "rotId needs to be smaller than 2",
            )


class CnnMeasurementTask(Task):
    """Measure CNN shapes and shear responses at given positions.

    Call :meth:`run` with the data dict prepared by
    ``AnacalTask.prepare_data`` for one band plus the cell the sources
    belong to.  Output columns, per source and prefixed by
    ``base_column_name``: ``cnn_e1``, ``cnn_de1_dg1``, ``cnn_de1_dg2``,
    ``cnn_e2``, ``cnn_de2_dg1``, ``cnn_de2_dg2`` (and with 4 tangents
    ``cnn_de{1,2}_dj{1,2}`` [1/arcsec]).  A source whose stamp does not
    fit in the cell, or whose ``n_mask_base`` exceeds
    ``n_mask_base_max``, gets NaN -- the rows stay aligned with the
    detection catalog.
    """

    _DefaultName = "CnnMeasurementTask"
    ConfigClass = CnnMeasurementConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, CnnMeasurementConfig)
        if not self.config.model_path:
            raise ValueError("CnnMeasurementTask: config.model_path is required")
        self.model = load_cnn_export(self.config.model_path)
        self.npix = self.model["stamp_size"]
        self.sigma_arcsec = (
            float(self.config.sigma_arcsec) if self.config.sigma_arcsec > 0 else self.model["sigma_arcsec"]
        )
        if not self.sigma_arcsec > 0:
            raise ValueError(
                "CnnMeasurementTask: no sigma_arcsec in the config nor in the model export"
            )
        # The runner keeps persistent work buffers, so the measurement
        # pipe's thread pool needs one per thread.
        self._local = threading.local()
        return

    def _net(self):
        net = getattr(self._local, "net", None)
        if net is None:
            cls = anacal.math.D4ShapeNet32 if self.config.precision == "float32" else anacal.math.D4ShapeNet64
            m = self.model
            net = cls(m["blocks"], m["head_e1"], m["head_e2"], m["wnorm"], m["wpool"], m["res"], 1e-5)
            self._local.net = net
        return net

    def column_names(self, base_column_name: str | None) -> list:
        prefix = base_column_name or ""
        names = []
        for comp in ("e1", "e2"):
            names += [f"{prefix}cnn_{comp}", f"{prefix}cnn_d{comp}_dg1", f"{prefix}cnn_d{comp}_dg2"]
            if self.config.tangents == 4:
                names += [f"{prefix}cnn_d{comp}_dj1", f"{prefix}cnn_d{comp}_dj2"]
        return names

    def run(
        self,
        *,
        pixel_scale: float,
        noise_variance: float,
        gal_array: NDArray,
        psf_array: NDArray,
        noise_array: NDArray | None,
        detection: NDArray,
        cells: list,
        noise2_array: NDArray | None = None,
        base_column_name: str | None = None,
        begin_x: int = 0,
        begin_y: int = 0,
        n_mask_base_max: float | None = None,
        renoise_mode: str = "rotate",
        renoise_c: float = 0.1,
        renoise_delta: float = 0.01,
        noise_corr: NDArray | None = None,
        noise_corrs: list | None = None,
        **kwargs,
    ) -> NDArray:
        """Measure the detections of ONE cell.

        Parameters
        ----------
        pixel_scale, noise_variance, gal_array, psf_array, noise_array,
        noise2_array, begin_x, begin_y, noise_corr, noise_corrs :
            As prepared by ``AnacalTask.prepare_data`` for this band
            (``noise2_array`` only in spectral mode; ``psf_array`` is the
            patch PSF, the cell's own stamp is taken from ``cells[0]``).
        detection : NDArray
            AnaCal detection catalog (``x1_det``, ``x2_det`` in arcsec,
            optionally ``n_mask_base``), the sources of this cell.
        cells : list
            One ``anacal.geometry.cell`` with its ``psf_array``.
        n_mask_base_max, renoise_mode, renoise_c, renoise_delta :
            Handed down by the measurement pipe, like for the FPFS task.
        """
        assert isinstance(self.config, CnnMeasurementConfig)
        if len(cells) != 1:
            raise ValueError("CnnMeasurementTask.run measures one cell at a time")
        cell = cells[0]
        if renoise_mode not in ("rotate", "spectral"):
            raise ValueError(f"renoise_mode must be 'rotate' or 'spectral', got {renoise_mode!r}")
        spectral = renoise_mode == "spectral" and noise_array is not None
        if spectral and noise2_array is None:
            raise ValueError("spectral renoising needs the second noise realization noise2_array")
        if noise_corr is None and noise_corrs is not None and len(noise_corrs) == 1:
            noise_corr = noise_corrs[0]

        names = self.column_names(base_column_name)
        out = np.full(len(detection), np.nan, dtype=np.dtype([(n, np.float64) for n in names]))
        if len(detection) == 0:
            return out

        npix = self.npix
        half = npix // 2
        xp = np.rint(np.asarray(detection["x1_det"]) / pixel_scale - begin_x).astype(int)
        yp = np.rint(np.asarray(detection["x2_det"]) / pixel_scale - begin_y).astype(int)
        x0 = xp - int(cell.xmin) - half
        y0 = yp - int(cell.ymin) - half
        inside = (x0 >= 0) & (y0 >= 0) & (x0 + npix <= int(cell.nx)) & (y0 + npix <= int(cell.ny))
        if n_mask_base_max is not None and "n_mask_base" in (detection.dtype.names or ()):
            inside &= np.asarray(detection["n_mask_base"]) <= n_mask_base_max
        if not np.any(inside):
            return out

        cell_psf = getattr(cell, "psf_array", None)
        if cell_psf is None or np.size(cell_psf) == 0:
            cell_psf = psf_array
        cell_psf = np.ascontiguousarray(cell_psf, dtype=np.float64)
        klim = anacal.task.KLIM_PIX / pixel_scale
        iq = anacal.image.ImageQ(int(cell.nx), int(cell.ny), pixel_scale, self.sigma_arcsec, klim, True)
        if spectral:
            spectrum = anacal.renoise.make_noise_spectrum(
                int(cell.nx),
                pixel_scale,
                float(noise_variance),
                None if noise_corr is None else np.ascontiguousarray(noise_corr, dtype=np.float64),
            )
            setup = anacal.renoise.Setup(
                c=renoise_c, spectra=[spectrum], noise_rotated=True, mode="spectral", delta=renoise_delta
            )
            rk = anacal.renoise.make_cell_kernels(setup, cell_psf, klim)[0]
            q = iq.prepare_qtensor(gal_array, cell_psf, cell.xcen, cell.ycen, noise_array, noise2_array, rk)
        else:
            q = iq.prepare_qtensor(gal_array, cell_psf, cell.xcen, cell.ycen, noise_array)
        q = q.reshape([1, int(cell.ny), int(cell.nx)])

        net = self._net()
        nc = self.config.tangents + 1
        for i in np.flatnonzero(inside):
            stamp = anacal.math.crop2d(q, int(y0[i]), int(x0[i]), npix, npix)
            stamp = stamp.astype(self.config.precision, self.config.tangents)
            res = np.asarray(net.measure(stamp), dtype=np.float64)  # (2, nc)
            row = out[i]
            for k, comp in enumerate(("e1", "e2")):
                vals = res[k]
                row[names[k * nc + 0]] = vals[0]
                row[names[k * nc + 1]] = vals[1]
                row[names[k * nc + 2]] = vals[2]
                if nc == 5:
                    row[names[k * nc + 3]] = vals[3]
                    row[names[k * nc + 4]] = vals[4]
        return out
