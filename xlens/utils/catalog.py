def getPatchInner(sources, patchInfo, pixel_scale):
    """Check if each source centroid is in the inner bbox of a patch.

    Uses ``patchInfo.getInnerBBox()`` from the **skymap** (not the
    exposure or MultipleCellCoadd). This is important because:

    - The skymap patch inner bbox defines the non-overlapping tiling of the
      sky. Each sky position belongs to exactly one patch's inner region.
    - Using the skymap inner bbox ensures correct deduplication across
      neighboring patches.

    Parameters
    ----------
    sources : structured array
        Catalog with ``x1``, ``x2`` columns in arcsec.
    patchInfo : `lsst.skymap.PatchInfo`
        Patch info from ``skyMap[tract][patch]``.
    pixel_scale : float
        Pixel scale in arcsec/pixel (to convert x1/x2 to pixel coords).

    Returns
    -------
    isPatchInner : array-like of `bool`
        ``True`` for each source whose centroid falls within the
        skymap's patch inner region.
    """
    from lsst.geom import Box2D

    # Use the skymap's patch inner bbox for deduplication.
    # Do NOT use MultipleCellCoadd.inner_bbox or exposure.getBBox(),
    # which cover the full outer extent including overlap regions.
    innerFloatBBox = Box2D(patchInfo.getInnerBBox())
    inInner = innerFloatBBox.contains(
        sources["x1"] / pixel_scale, sources["x2"] / pixel_scale
    )
    return inInner


def getTractInner(sources, tractInfo, skyMap):
    """Set a flag for each source that the skyMap includes in tractInfo.

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog with pre-calculated centroids.
    tractInfo : `lsst.skymap.TractInfo`
        Tract object
    skyMap : `lsst.skymap.BaseSkyMap`
        Sky tessellation object

    Returns
    -------
    isTractInner : array-like of `bool`
        True if the skyMap.findTract method returns
        the same tract as tractInfo.
    """
    isTractInner = (
        skyMap.findTractIdArray(
            sources["ra"],
            sources["dec"],
            degrees=True,
        )
        == tractInfo.getId()
    )
    return isTractInner


def set_isPrimary(sources, skyMap, tractInfo, patchInfo, pixel_scale):
    """Set isPrimary and related flags on sources.

    For coadded imaging, the `isPrimary` flag returns True when an object is in
    the inner region of a coadd patch, is in the inner region of a coadd tract

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceTable. Reads in centroid fields and an nChild field.
        Writes is-patch-inner, is-tract-inner, and is-primary flags.
    skyMap : `lsst.skymap.BaseSkyMap`
        Sky tessellation object
    tractInfo : `lsst.skymap.TractInfo`, optional
        Tract object; required if ``self.isSingleFrame`` is False.
    patchInfo : `lsst.skymap.PatchInfo`
        Patch object; required if ``self.isSingleFrame`` is False.
    pixel_scale: `float`
        pixel scale
    """
    # Mark whether sources are contained within the inner regions of the
    # given tract/patch
    isPatchInner = getPatchInner(sources, patchInfo, pixel_scale)
    isTractInner = getTractInner(sources, tractInfo, skyMap)
    sources["is_primary"] = isTractInner & isPatchInner
    return
