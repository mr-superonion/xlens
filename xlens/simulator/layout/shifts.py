import numpy as np


def get_grid_shifts(*, rng, dim, pixel_scale, spacing):
    """
    get a set of gridded shifts, with random shifts at the pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    pixel_scale: float
        pixel scale
    spacing: float
        Spacing of the lattice

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    width = dim * pixel_scale
    n_on_side = int(dim / spacing * pixel_scale)

    ntot = n_on_side**2

    # ix/iy are really on the sky
    grid = spacing*(np.arange(n_on_side) - (n_on_side-1)/2)

    shifts = np.zeros(ntot, dtype=[('dx', 'f8'), ('dy', 'f8')])

    i = 0
    for ix in range(n_on_side):
        for iy in range(n_on_side):
            dx = grid[ix] + pixel_scale * rng.uniform(low=-0.5, high=0.5)
            dy = grid[iy] + pixel_scale * rng.uniform(low=-0.5, high=0.5)

            shifts['dx'][i] = dx
            shifts['dy'][i] = dy
            i += 1

    pos_bounds = (-width/2, width/2)
    msk = (
        (shifts['dx'] >= pos_bounds[0])
        & (shifts['dx'] <= pos_bounds[1])
        & (shifts['dy'] >= pos_bounds[0])
        & (shifts['dy'] <= pos_bounds[1])
    )
    shifts = shifts[msk]
    return shifts


def get_hex_shifts(*, rng, dim, pixel_scale, spacing):
    """
    get a set of hex grid shifts, with random shifts at the pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    pixel_scale: float
        pixel scale
    spacing: float
        Spacing of the hexagonal lattice

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """
    from hexalattice.hexalattice import create_hex_grid

    width = dim * pixel_scale
    n_on_side = int(width / spacing) + 1

    nx = int(n_on_side * np.sqrt(2))
    # the factor of 0.866 makes sure the grid is square-ish
    ny = int(n_on_side * np.sqrt(2) / 0.8660254)

    # here the spacing between grid centers is 1
    hg, _ = create_hex_grid(nx=nx, ny=ny, rotate_deg=rng.uniform() * 360)

    # convert the spacing to right number of pixels
    # we also recenter the grid since it comes out centered at 0,0
    hg *= spacing
    upos = hg[:, 0].ravel()
    vpos = hg[:, 1].ravel()

    # dither
    upos += pixel_scale * rng.uniform(low=-0.5, high=0.5, size=upos.shape[0])
    vpos += pixel_scale * rng.uniform(low=-0.5, high=0.5, size=vpos.shape[0])

    pos_bounds = (-width/2, width/2)
    msk = (
        (upos >= pos_bounds[0])
        & (upos <= pos_bounds[1])
        & (vpos >= pos_bounds[0])
        & (vpos <= pos_bounds[1])
    )
    upos = upos[msk]
    vpos = vpos[msk]

    ntot = upos.shape[0]
    shifts = np.zeros(ntot, dtype=[('dx', 'f8'), ('dy', 'f8')])
    shifts["dx"] = upos
    shifts["dy"] = vpos
    return shifts


def get_random_shifts(*, rng, dim, pixel_scale, size):
    """
    get a set of random shifts in a square, with random shifts at the pixel
    scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    pixel_scale: float
        pixel scale
    size: int
        Number of objects to draw.

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    halfwidth = dim / 2.0
    if halfwidth < 0:
        raise ValueError("dim < 0")

    low = -halfwidth * pixel_scale
    high = halfwidth * pixel_scale

    shifts = np.zeros(size, dtype=[('dx', 'f8'), ('dy', 'f8')])

    shifts['dx'] = rng.uniform(low=low, high=high, size=size)
    shifts['dy'] = rng.uniform(low=low, high=high, size=size)
    return shifts


def get_random_disk_shifts(*, rng, dim, pixel_scale, size):
    """Gets a set of random shifts on a disk, with random shifts at the
    pixel scale

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    dim: int
        Dimensions of the final image
    pixel_scale: float
        pixel scale
    size: int
        Number of objects to draw.

    Returns
    -------
    shifts: array
        Array with dx, dy offset fields for each point, in
        arcsec
    """

    radius = dim / 2.0 * pixel_scale
    if radius < 0:
        raise ValueError("radius < 0")
    radius_square = radius**2.

    # evenly distributed within a radius, min(nx, ny)*rfrac
    rarray = np.sqrt(radius_square*rng.rand(size))   # radius
    tarray = rng.uniform(0., 2*np.pi, size)   # theta (0, pi/nrot)

    shifts = np.zeros(size, dtype=[('dx', 'f8'), ('dy', 'f8')])
    shifts['dx'] = rarray*np.cos(tarray)
    shifts['dy'] = rarray*np.sin(tarray)
    return shifts
