from lsst.geom import Box2D


def getPatchInner(sources, patchInfo, pixel_scale):
    """Set a flag for each source if it is in the innerBBox of a patch.

    Parameters
    ----------
    sources : `lsst.afw.table.SourceCatalog`
        A sourceCatalog with pre-calculated centroids.
    patchInfo : `lsst.skymap.PatchInfo`
        Information about a `SkyMap` `Patch`.

    Returns
    -------
    isPatchInner : array-like of `bool`
        `True` for each source that has a centroid
        in the inner region of a patch.
    """
    # set inner flags for each source and set primary flags for
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
