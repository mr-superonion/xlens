import galsim
from astropy.cosmology import Planck18
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from descwl_shear_sims.constants import WORLD_ORIGIN
from descwl_shear_sims.shear import _get_shear_res_dict
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u



class ShearHalo(object):
    def __init__(
        self,
        mass,
        conc,
        z_lens,
        ra_lens=200.0,
        dec_lens=0.0,
        halo_profile="NFW",
        cosmo=None,
        no_kappa=False,
    ):
        """Shear distortion from halo

        Args:
        mass (float):               mass of the halo [M_sun]
        conc (float):               concerntration
        z_lens (float):             lens redshift
        ra_lens (float):            ra of halo position [arcsec]
        dec_lens (float):           dec of halo position [arcsec]
        halo_profile (str):         halo profile name
        cosmo (astropy.cosmology):  cosmology object
        no_kappa (bool):            if True, turn off kappa field
        """
        
        if ra_lens != 200.0 or dec_lens != 0.:
            raise ValueError(
                f"ra_lens and dec_lens should be 200.0 and 0.0 respectively, ra: {ra_lens}, dec: {dec_lens}"
            )

        if cosmo is None:
            cosmo = Planck18
        self.cosmo = cosmo
        self.mass = mass
        self.z_lens = z_lens
        self.conc = conc
        self.no_kappa = no_kappa
        self.lens = LensModel(lens_model_list=[halo_profile])
        self.pos_lens = galsim.PositionD(ra_lens * 3600., dec_lens * 3600.)
        self.lens_eqn_solver = LensEquationSolver(lensModel=self.lens)
        return

    # def get_offset(ra_arcsec, dec_arcsec, ra0_arcsec, dec0_arcsec):
    #     c0 = SkyCoord(ra=ra0_arcsec * u.arcsec, dec=dec0_arcsec * u.arcsec)
    #     c1 = SkyCoord(ra=ra_arcsec * u.arcsec, dec=dec_arcsec * u.arcsec)
    #     dlon, dlat = c0.spherical_offsets_to(c1)
    #     delta_ra_astro = dlon.to(u.arcsec).value
    #     delta_dec_astro = dlat.to(u.arcsec).value

    #     return delta_ra_astro, delta_dec_astro

    def distort_galaxy(self, gso, shift, redshift):
        """This function distorts the galaxy's shape and position
        Parameters
        ---------
        gso (galsim object):        galsim galaxy
        shift (galsim.PositionD):   position of the galaxy
        redshift (float):           redshift of galaxy

        Returns
        ---------
        gso, shift:
            distorted galaxy object and shift
        """
        
        # assert WORLD_ORIGIN.ra == 200.0 and WORLD_ORIGIN.dec == 0.0, \
        #     "WORLD_ORIGIN should be at (200.0, 0.0) in this code"
        
        # world_pos = WORLD_ORIGIN.deproject(shift.x * galsim.arcsec,
        #                                  shift.y * galsim.arcsec)
        # ra = world_pos.ra / galsim.arcsec
        # dec = world_pos.dec / galsim.arcsec
        # source_pos = galsim.PositionD(ra, dec)

        
        if redshift > self.z_lens:
            # r = source_pos - self.pos_lens
            # delta_ra, delta_dec = get_offset(
            #     ra_arcsec=source_pos.x,
            #     dec_arcsec=source_pos.y,
            #     ra0_arcsec=self.pos_lens.x,
            #     dec0_arcsec=self.pos_lens.y,
            # )

            r = shift
            
            lens_cosmo = LensCosmo(
                z_lens=self.z_lens,
                z_source=redshift,
                cosmo=self.cosmo,
            )
            rs_angle, alpha_rs = lens_cosmo.nfw_physical2angle(
                M=self.mass, c=self.conc
            )
            kwargs = [{"Rs": rs_angle, "alpha_Rs": alpha_rs}]

            lensed_ra, lensed_dec = self.lens_eqn_solver.image_position_from_source(
                r.x,
                r.y,
                kwargs,
                min_distance=0.2,  # chance of finding solutions as close as min_distance away from each other
                search_window=50,  # largest distance to find a solution for
            )

            # if lenstronomy cannot find a solution
            # do not shift
            if len(lensed_ra) == 0 or len(lensed_dec) == 0:
                lensed_ra, lensed_dec = shift.x, shift.y
            else:
                lensed_ra, lensed_dec = lensed_ra[0], lensed_dec[0]
                
            f_xx, f_xy, f_yx, f_yy = self.lens.hessian(
                lensed_ra, lensed_dec, kwargs
            )
            gamma1 = 1.0 / 2 * (f_xx - f_yy)
            gamma2 = f_xy
            if self.no_kappa:
                kappa = 0.0
            else:
                kappa = 1.0 / 2 * (f_xx + f_yy)

            g1 = gamma1 / (1 - kappa)
            g2 = gamma2 / (1 - kappa)
            mu = 1.0 / ((1 - kappa) ** 2 - gamma1**2 - gamma2**2)

            if g1**2.0 + g2**2.0 > 0.95:
                return _get_shear_res_dict(gso, shift, gamma1, gamma2, kappa)
                return gso, shift, shift, gamma1, gamma2, kappa

            # dra, ddec = self.lens.alpha(lensed_ra, lensed_dec, kwargs)
            # dra, ddec = lensed_ra - shift.x, lensed_dec - shift.y
            gso = gso.lens(g1=g1, g2=g2, mu=mu)
            # lensed_shift = shift + galsim.PositionD(dra, ddec)
            
            # lensed_shift = WORLD_ORIGIN.project(
            #     galsim.CelestialCoord((lensed_ra + 200 * 3600) * galsim.arcsec, lensed_dec * galsim.arcsec)
            # )
            # lensed_shift = galsim.PositionD(lensed_shift[0] / galsim.arcsec, lensed_shift[1] / galsim.arcsec)

            lensed_shift = galsim.PositionD(lensed_ra, lensed_dec)
            
            # lensed_shift = galsim.PositionD(lensed_ra, lensed_dec)
            # print(f"shift: {shift}, lensed_shift: {lensed_shift}, gso: {gso}, gamma1: {gamma1}, gamma2: {gamma2}, kappa: {kappa}, mass: {self.mass}, conc: {self.conc}")
            return _get_shear_res_dict(gso, lensed_shift, gamma1, gamma2, kappa)