"""CnnMeasurementTask: the model export loader and weight maps, and the
forced CNN measurement of one cell in both renoising modes, on the
synthetic scene of test_anacal_renoise."""

import json

import anacal
import numpy as np
import pytest

from xlens.processor.cnn import (
    CnnMeasurementConfig,
    CnnMeasurementTask,
    gaussian_weight_map,
    load_cnn_export,
)
from xlens.utils.image.noise import generate_filler_noise

from test_anacal_renoise import MAG_ZERO, SCALE, _psf, _scene

STAMP = 32


def write_tiny_export(path, *, channels=4, nlayer=2, hidden=6, n=STAMP, shared_head=False, seed=5):
    """A random Forward8_fixW_CNN in the cpu_src .npz export format."""
    rng = np.random.default_rng(seed)
    arrays = {}
    for i in range(nlayer):
        cin = 1 if i == 0 else channels
        arrays[f"blocks.{i}.conv.weight"] = rng.normal(0.0, (2.0 / (9 * cin)) ** 0.5, (channels, cin, 3, 3))
        arrays[f"blocks.{i}.conv.bias"] = rng.normal(0.0, 0.1, channels)
        arrays[f"blocks.{i}.norm.weight"] = 1.0 + rng.normal(0.0, 0.1, (1, channels, 1, 1))
        arrays[f"blocks.{i}.norm.bias"] = rng.normal(0.0, 0.1, (1, channels, 1, 1))
        if cin != channels:
            arrays[f"blocks.{i}.skip.weight"] = rng.normal(0.0, 1.0, (channels, cin, 1, 1))
    heads = ["head_shared"] if shared_head else ["head_e1", "head_e2"]
    for key in heads:
        arrays[f"{key}.net.1.weight"] = rng.normal(0.0, channels**-0.5, (hidden, channels))
        arrays[f"{key}.net.3.weight"] = rng.normal(0.0, hidden**-0.5, (hidden, hidden))
        arrays[f"{key}.net.5.weight"] = rng.normal(0.0, hidden**-0.5, (hidden, hidden))
        arrays[f"{key}.net.7.weight"] = rng.normal(0.0, hidden**-0.5, (1, hidden))
    meta = {
        "architecture": "Forward8_fixW_CNN",
        "format_version": 1,
        "model": {
            "base_channels": channels,
            "num_layers": nlayer,
            "res_factor": 0.1,
            "nl_sigma": 4.0,
            "gw_sigma": 3.0,
            "shared_head": shared_head,
            "suppress": True,
            "head_hidden": hidden,
            "in_dim": 1,
            "out_dim": 2,
        },
        "preprocessing": {"pixel_scale": 0.2, "sigma_arcsec": 0.45, "flim": 10.0, "klim": 50.0},
        "stamp_size": n,
    }
    arrays = {k: v.astype(np.float32) for k, v in arrays.items()}
    arrays["metadata"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    np.savez(path, **arrays)
    return path


def test_gaussian_weight_map_matches_definition():
    torch = pytest.importorskip("torch")
    n, sigma = 16, 4.0
    got = gaussian_weight_map(n, sigma)
    # gaussian_weight_2d of wl_D4T_CNN, written out
    y = torch.arange(n, dtype=torch.float64) - (n - 1) / 2
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    g = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2) / (2 * np.pi * sigma**2)
    g = g * torch.exp(-((xx**2 + yy**2) / 24.0**2) ** 4)
    g = (g + g.flip([0, 1]) + g.flip([0]) + g.flip([1])) / 4
    np.testing.assert_allclose(got, g.numpy(), rtol=1e-12)
    assert np.argmax(got) == (n // 2) * n + n // 2 or got.max() == got[n // 2 - 1, n // 2 - 1]


def test_load_export(tmp_path):
    path = write_tiny_export(str(tmp_path / "tiny.npz"))
    m = load_cnn_export(path)
    assert m["stamp_size"] == STAMP and m["sigma_arcsec"] == 0.45 and m["res"] == 0.1
    assert len(m["blocks"]) == 2 and m["blocks"][0][4] is not None and m["blocks"][1][4] is None
    assert len(m["head_e1"]) == 4 and m["head_e1"][0].shape == (6, 4)
    assert m["wnorm"].shape == (STAMP, STAMP)
    shared = write_tiny_export(str(tmp_path / "shared.npz"), shared_head=True)
    ms = load_cnn_export(shared)
    assert ms["head_e1"][0] is ms["head_e2"][0]
    # a plain array file is refused
    np.savez(str(tmp_path / "bad.npz"), a=np.zeros(3))
    with pytest.raises(ValueError, match="metadata"):
        load_cnn_export(str(tmp_path / "bad.npz"))


def _cells(n):
    return anacal.geometry.get_cell_list(
        img_nx=n, img_ny=n, cell_nx=256, cell_ny=256, cell_overlap=64, scale=SCALE
    )


def _detections(cell, ngrid=64, nstamp=8):
    """The grid galaxies of the scene owned by this cell (x1_det, x2_det
    in arcsec, like the AnaCal catalog), plus one source too close to the
    cell edge for a stamp."""
    xs, ys = [], []
    for j in range(nstamp):
        for i in range(nstamp):
            x, y = i * ngrid + ngrid // 2, j * ngrid + ngrid // 2
            if cell.xmin_in <= x < cell.xmax_in and cell.ymin_in <= y < cell.ymax_in:
                xs.append(x)
                ys.append(y)
    xs.append(cell.xmin + 3)  # stamp would stick out of the cell
    ys.append(cell.ymin_in + 5)
    det = np.zeros(len(xs), dtype=[("x1_det", "f8"), ("x2_det", "f8"), ("n_mask_base", "f4")])
    det["x1_det"] = np.array(xs) * SCALE
    det["x2_det"] = np.array(ys) * SCALE
    return det


@pytest.mark.parametrize("renoise_mode", ["rotate", "spectral"])
def test_task_measures_one_cell(tmp_path, renoise_mode):
    psf = _psf()
    img, noise, var, n = _scene(psf)
    cells = _cells(n)
    cell = cells[0]
    det = _detections(cell)
    det["n_mask_base"][-2] = 0.5  # masked beyond the cut
    noise2 = None
    if renoise_mode == "spectral":
        noise2 = generate_filler_noise(
            ny=n, nx=n, pixel_scale=SCALE, seed=7, band="i", noise_variance=var, noise_corr=None
        ).astype(np.float32)
    config = CnnMeasurementConfig()
    config.model_path = write_tiny_export(str(tmp_path / "tiny.npz"))
    task = CnnMeasurementTask(config=config)
    assert task.sigma_arcsec == 0.45  # from the export
    cat = task.run(
        pixel_scale=SCALE,
        mag_zero=MAG_ZERO,
        noise_variance=var,
        gal_array=img,
        psf_array=psf,
        mask_array=np.zeros(img.shape, dtype=np.uint8),
        noise_array=noise,
        noise2_array=noise2,
        detection=det,
        cells=[cell],
        base_column_name="lsst_i_",
        n_mask_base_max=0.1,
        renoise_mode=renoise_mode,
    )
    assert len(cat) == len(det)
    assert cat.dtype.names == (
        "lsst_i_cnn_e1", "lsst_i_cnn_de1_dg1", "lsst_i_cnn_de1_dg2",
        "lsst_i_cnn_e2", "lsst_i_cnn_de2_dg1", "lsst_i_cnn_de2_dg2",
    )
    e1 = cat["lsst_i_cnn_e1"]
    # the edge source and the masked one are NaN, every other row is measured
    assert np.isnan(e1[-1]) and np.isnan(e1[-2])
    assert np.all(np.isfinite(e1[:-2])) and len(e1) > 4
    assert np.all(np.isfinite(cat["lsst_i_cnn_de1_dg1"][:-2]))
    # identical galaxies on the grid: the shapes scatter only by the noise
    assert np.std(e1[:-2]) < 0.5 * max(1e-3, abs(np.mean(e1[:-2])) + 0.1)

    # one source by hand through ImageQ + D4ShapeNet: the same numbers
    has_psf = np.size(getattr(cell, "psf_array", [])) > 0
    cell_psf = np.ascontiguousarray(cell.psf_array if has_psf else psf)
    klim = anacal.task.KLIM_PIX / SCALE
    iq = anacal.image.ImageQ(int(cell.nx), int(cell.ny), SCALE, 0.45, klim, True)
    if renoise_mode == "spectral":
        spectrum = anacal.renoise.make_noise_spectrum(int(cell.nx), SCALE, var, None)
        setup = anacal.renoise.Setup(
            c=0.1, spectra=[spectrum], noise_rotated=True, mode="spectral", delta=0.01
        )
        rk = anacal.renoise.make_cell_kernels(setup, cell_psf, klim)[0]
        q = iq.prepare_qtensor(img, cell_psf, cell.xcen, cell.ycen, noise, noise2, rk)
    else:
        q = iq.prepare_qtensor(img, cell_psf, cell.xcen, cell.ycen, noise)
    q = q.reshape([1, int(cell.ny), int(cell.nx)])
    xp = int(round(det["x1_det"][0] / SCALE)) - int(cell.xmin) - STAMP // 2
    yp = int(round(det["x2_det"][0] / SCALE)) - int(cell.ymin) - STAMP // 2
    stamp = anacal.math.crop2d(q, yp, xp, STAMP, STAMP).astype("float32", 2)
    m = task.model
    net = anacal.math.D4ShapeNet32(
        m["blocks"], m["head_e1"], m["head_e2"], m["wnorm"], m["wpool"], m["res"], 1e-5
    )
    ref = net.measure(stamp)
    np.testing.assert_allclose(
        [cat["lsst_i_cnn_e1"][0], cat["lsst_i_cnn_de1_dg1"][0], cat["lsst_i_cnn_de1_dg2"][0]],
        ref[0], rtol=1e-6, atol=1e-7,
    )


def test_task_config_checks(tmp_path):
    config = CnnMeasurementConfig()
    with pytest.raises(ValueError, match="model_path"):
        CnnMeasurementTask(config=config)
    config.model_path = write_tiny_export(str(tmp_path / "tiny.npz"))
    config.sigma_arcsec = 0.3
    task = CnnMeasurementTask(config=config)
    assert task.sigma_arcsec == 0.3  # the config overrides the export
    config.precision = "float16"
    with pytest.raises(Exception, match="precision"):
        config.validate()
