"""Log-normal convergence and shear field on a flat sky."""

import galsim
import numpy as np
import scipy.interpolate


class ShearLogNormalFlat:

    """Flat-sky log-normal shear field generated from a CCL weak-lensing
    power spectrum.

    The constructor computes kappa, gamma1, and gamma2 maps and builds
    bivariate spline interpolators so that ``distort_galaxy`` can evaluate
    the lensing fields at arbitrary sky positions.

    Parameters
    ----------
    z_source : float
        Source redshift for the weak-lensing tracer.
    field_size_deg : float, optional
        Side length of the square field in degrees.
    npix : int, optional
        Number of pixels per side used for the Fourier-space generation.
    seed : int or None, optional
        Random seed for reproducible field realisations.
    """

    def __init__(
        self, z_source, field_size_deg=5.0, npix=2048, seed=None,
    ):
        import pyccl as ccl

        # Default to use planck 2018 Cosmology
        h, Obh2, Och2 = 0.6736, 0.02237, 0.1200
        self.cosmo = ccl.Cosmology(
            Omega_b=Obh2/h**2, Omega_c=Och2/h**2, h=h,
            n_s=0.9649, A_s=2.100549e-9,  # k=0.05 Mpc^-1
            Neff=3.046, m_nu=0.06,
        )
        self.z_source = z_source
        self.field_size_deg = field_size_deg
        self.npix = npix
        self.seed = seed
        np.random.seed(self.seed)

        # 1. Get Power Spectrum
        # Create a narrow redshift distribution for the WeakLensingTracer
        z_arr = np.linspace(0, 2 * self.z_source, 100)
        dndz_arr = np.exp(-(z_arr - self.z_source)**2 / (2 * 0.01**2))
        dndz_arr /= np.trapz(dndz_arr, z_arr)

        tracer = ccl.WeakLensingTracer(self.cosmo, dndz=(z_arr, dndz_arr))

        # Define ell range for power spectrum calculation
        field_size_rad = np.deg2rad(self.field_size_deg)
        pixel_scale_rad = field_size_rad / self.npix
        l_min = 2 * np.pi / field_size_rad
        l_max = np.pi / pixel_scale_rad
        ell = np.logspace(np.log10(l_min), np.log10(l_max), 1024)
        cl_kappa = ccl.angular_cl(self.cosmo, tracer, tracer, ell)
        self.ell = ell
        self.cl_kappa = cl_kappa

        # 2. Generate Gaussian Field (Fourier Space)
        # Create k-space grid
        k_modes = np.fft.fftfreq(self.npix, d=pixel_scale_rad) * 2 * np.pi
        kx, ky = np.meshgrid(k_modes, k_modes)
        k_abs = np.sqrt(kx**2 + ky**2)

        # Interpolate power spectrum to k_abs grid
        cl_interp = np.interp(k_abs, ell, cl_kappa, left=0.0, right=0.0)

        # Scale power spectrum for grid and generate field The variance of the
        # Fourier modes must be scaled to account for grid size and FFT
        # normalization to ensure the real-space map has the correct variance.
        # Var(FFT(kappa)) = (N_pix^4 / L^2) * C_l, where L is field size in
        # radians. A factor of 2 is needed to compensate for power loss when
        # taking .real of ifft of non-hermitian field.
        pk_2d_scaled = 2 * cl_interp * (self.npix**4 / (field_size_rad**2))

        # Generate Gaussian random field in Fourier space
        # The 1/sqrt(2) is for the complex random numbers, which have variance
        # 1 in real and imag parts.
        gaussian_field_fourier = np.sqrt(pk_2d_scaled) * (
            np.random.normal(size=(self.npix, self.npix))
            + 1j * np.random.normal(size=(self.npix, self.npix))
        ) / np.sqrt(2.0)

        # 3. Transform to Real Space
        kappa_G = np.fft.ifft2(gaussian_field_fourier).real
        self.kappa_G_map = kappa_G

        # 4. Apply Log-Normal Transform
        sigma_G_squared = np.var(kappa_G)
        kappa_LN = np.exp(kappa_G - sigma_G_squared / 2) - 1
        self.kappa_LN_map = kappa_LN

        # 5. Generate Shear Fields
        kappa_LN_fourier = np.fft.fft2(kappa_LN)

        # Avoid division by zero at k_abs = 0
        k_abs[k_abs == 0] = 1e-10

        gamma1_fourier = ((kx**2 - ky**2) / k_abs**2) * kappa_LN_fourier
        gamma2_fourier = - ((2 * kx * ky) / k_abs**2) * kappa_LN_fourier

        # 6. Transform Shear to Real Space
        self.gamma1_map = np.fft.ifft2(gamma1_fourier).real
        self.gamma2_map = np.fft.ifft2(gamma2_fourier).real
        self.kappa_map = kappa_LN

        # 7. Create Interpolators
        x_coords = np.linspace(
            -self.field_size_deg / 2, self.field_size_deg / 2, self.npix,
        )
        y_coords = np.linspace(
            -self.field_size_deg / 2, self.field_size_deg / 2, self.npix,
        )

        self.kappa_interp = scipy.interpolate.RectBivariateSpline(
            x_coords, y_coords, self.kappa_map
        )
        self.gamma1_interp = scipy.interpolate.RectBivariateSpline(
            x_coords, y_coords, self.gamma1_map
        )
        self.gamma2_interp = scipy.interpolate.RectBivariateSpline(
            x_coords, y_coords, self.gamma2_map
        )

    def distort_galaxy(self, src):
        """Return lensing distortions interpolated at the galaxy position.

        Parameters
        ----------
        src : numpy structured scalar
            Row with ``"dx"`` and ``"dy"`` fields (arcseconds).

        Returns
        -------
        dict
            Standard shear result dict (see :func:`_get_shear_res_dict`).
        """
        from .utils import _get_shear_res_dict
        dx_deg = src["dx"] / 3600.0
        dy_deg = src["dy"] / 3600.0
        kappa = self.kappa_interp(dx_deg, dy_deg)[0, 0]
        gamma1 = self.gamma1_interp(dx_deg, dy_deg)[0, 0]
        gamma2 = self.gamma2_interp(dx_deg, dy_deg)[0, 0]

        g1 = gamma1 / (1 - kappa)
        g2 = gamma2 / (1 - kappa)
        mu = 1.0 / ((1 - kappa) ** 2 - gamma1**2 - gamma2**2)

        mat = galsim.Shear(g1=g1, g2=g2).getMatrix() * np.sqrt(mu)
        lensed_x = src["dx"] * mat[0, 0] + src["dy"] * mat[0, 1]
        lensed_y = src["dx"] * mat[1, 0] + src["dy"] * mat[1, 1]

        return _get_shear_res_dict(
            lensed_x=lensed_x,
            lensed_y=lensed_y,
            gamma1=gamma1,
            gamma2=gamma2,
            kappa=kappa,
            has_finite_shear=True
        )
