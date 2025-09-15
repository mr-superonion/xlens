import math
import lsst
import numpy as np

from .shifts import (
    get_grid_shifts,
    get_hex_shifts,
    get_random_disk_shifts,
    get_random_shifts,
)

GRID_SPACING = 11.0      # arcsec
HEX_SPACING = 11.0       # arcsec
RANDOM_DENSITY = 80.0    # per arcmin^2


class Layout:
    """
    Generate object positions on a coadd (flat sky), returning absolute
    **arcsec** coordinates in a structured array with fields ('dx','dy').

    Workflow
    --------
    1) Read center in pixels from `boundary_box.getCenterX/Y()` and convert to
       arcseconds via `wcs.getPixelScale().asArcseconds()`.
    2) Build a square sampling region based on the bbox, padded by 20″ on each
    side.
    3) Call the helper function to get **center-relative arcsec** shifts.
    4) Add the (arcsec) center to make **absolute** arcsec coordinates.

    Parameters
    ----------
    layout_name : {'grid','hex','random','random_disk'}
        Layout pattern for placing objects.
    wcs : SkyWcs-like
        Used to obtain pixel scale via `getPixelScale().asArcseconds()`.
    boundary_box : lsst.geom.Box2I or Box2D
        Pixel bounding box defining the coadd region.
    pad_arcsec: float, optional
        padding (arcsec) in x and y, which can be negative; defaults to 20
    sep_arcsec : float or None, optional
        Spacing (arcsec) for 'grid'/'hex'; defaults to
        GRID_SPACING/HEX_SPACING.

    Notes
    -----
    - All helper functions are expected to accept a `numpy.random.RandomState`
      and to return a structured array with fields ('dx','dy') in
      **arcseconds**, relative to the center of boundary box.
    - Returned `dx, dy` are **arcseconds**, absolute on the coadd tangent
      plane.
    """

    def __init__(
        self,
        *,
        layout_name: str,
        wcs: lsst.afw.geom.SkyWcs,
        boundary_box: lsst.geom.Box2I,
        pad_arcsec: float = 20.0,
        sep_arcsec: float | None = None,
    ):
        self.sep = sep_arcsec
        # Pixel scale (arcsec/pixel)
        pixel_scale_arcsec = float(wcs.getPixelScale().asArcseconds())

        # Box geometry (pixels)
        width = float(boundary_box.getWidth())
        height = float(boundary_box.getHeight())

        # Coadd center in pixels (legacy getters), then convert to arcsec
        x_center_pix = float(boundary_box.getCenterX())
        y_center_pix = float(boundary_box.getCenterY())
        self._x_center_arcsec = x_center_pix * pixel_scale_arcsec
        self._y_center_arcsec = y_center_pix * pixel_scale_arcsec

        # Square dimension with 20″ padding on each side
        pad_pix = pad_arcsec / pixel_scale_arcsec
        dim_pix = max(width, height) + 2.0 * pad_pix
        self._dim_pixels = int(math.ceil(dim_pix))

        self._pixscale_arcsec = pixel_scale_arcsec
        self._name = layout_name

        # Precompute area (arcmin^2) for Poisson mean when needed
        if layout_name in ("random", "random_disk"):
            if layout_name == "random":
                side_arcmin = (self._dim_pixels * self._pixscale_arcsec) / 60.0
                # ensure tiny positive area for very small boxes
                self._area_arcmin2 = max(
                    side_arcmin**2, (2.0 * self._pixscale_arcsec / 60.0) ** 2
                )
            else:
                radius_arcmin = max(
                    (self._dim_pixels * 0.5 * self._pixscale_arcsec) / 60.0,
                    (2.0 * self._pixscale_arcsec / 60.0)
                )
                self._area_arcmin2 = math.pi * radius_arcmin**2
        else:
            self._area_arcmin2 = 0.0


    @property
    def pixel_scale_arcsec(self) -> float:
        """Arcseconds per pixel."""
        return self._pixscale_arcsec

    @property
    def dim_pixels(self) -> int:
        """Square dimension (pixels) used for layout generation (includes
        padding)."""
        return self._dim_pixels

    @property
    def area_arcmin2(self) -> float:
        """Area used for Poisson draws (/arcmin^2). Zero for 'grid'/'hex'."""
        return self._area_arcmin2

    # ---------- Main API ----------

    def get_shifts(
        self,
        *,
        rng: np.random.RandomState,
        density: float = RANDOM_DENSITY,
    ) -> np.ndarray:
        """
        Generate absolute coadd-plane positions
        (dtype=[('dx','f8'),('dy','f8')], arcsec).

        Parameters
        ----------
        rng : numpy.random.RandomState
            Random number generator (old NumPy RNG API).
        density : float, optional
            Number density (/arcmin^2) for 'random'/'random_disk'. Ignored for
            'grid'/'hex'.

        Returns
        -------
        shifts : np.ndarray
            Structured array with fields ("dx","dy"), **arcseconds**, absolute.
        """
        sep = self.sep
        if not isinstance(rng, np.random.RandomState):
            raise TypeError(
                "rng must be numpy.random.RandomState (old generator)"
            )

        if self._name == "grid":
            spacing = float(sep if sep is not None else GRID_SPACING)
            shifts = get_grid_shifts(
                rng=rng,
                dim=self._dim_pixels,
                pixel_scale=self._pixscale_arcsec,
                spacing=spacing,
            )

        elif self._name == "hex":
            spacing = float(sep if sep is not None else HEX_SPACING)
            shifts = get_hex_shifts(
                rng=rng,
                dim=self._dim_pixels,
                pixel_scale=self._pixscale_arcsec,
                spacing=spacing,
            )

        elif self._name in ("random", "random_disk"):
            if self._area_arcmin2 <= 0.0:
                raise ValueError(f"Non-positive area for layout '{self._name}'")

            lam = max(self._area_arcmin2 * max(density, 0.0), 0.0)
            nobj = int(rng.poisson(lam))

            if self._name == "random":
                shifts = get_random_shifts(
                    rng=rng,
                    dim=self._dim_pixels,
                    pixel_scale=self._pixscale_arcsec,
                    size=nobj,
                )
            else:
                shifts = get_random_disk_shifts(
                    rng=rng,
                    dim=self._dim_pixels,
                    pixel_scale=self._pixscale_arcsec,
                    size=nobj,
                )

        else:
            raise ValueError(f"Unknown layout_name '{self._name}'")

        # center-relative (arcsec) -> absolute (arcsec)
        shifts["dx"] += self._x_center_arcsec
        shifts["dy"] += self._y_center_arcsec
        return shifts
