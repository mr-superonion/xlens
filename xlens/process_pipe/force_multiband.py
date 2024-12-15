# This file is part of drp_tasks.
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

import lsst.afw.table
import lsst.pex.config
import lsst.pipe.base as pipeBase
from lsst.meas.base._id_generator import SkyMapIdGeneratorConfig
from lsst.meas.base.applyApCorr import ApplyApCorrTask
from lsst.meas.base.catalogCalculation import CatalogCalculationTask
from lsst.meas.base.forcedMeasurement import ForcedMeasurementTask
from lsst.meas.extensions.scarlet.io import updateCatalogFootprints

__all__ = ("ForcedPhotCoaddConfig", "ForcedPhotCoaddTask")


class ForcedPhotCoaddConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("band", "skymap", "tract", "patch"),
    defaultTemplates={"inputCoaddName": "deep", "outputCoaddName": "deep"},
):
    inputSchema = pipeBase.connectionTypes.InitInput(
        doc="Schema for the input measurement catalogs.",
        name="{inputCoaddName}Coadd_ref_schema",
        storageClass="SourceCatalog",
    )
    outputSchema = pipeBase.connectionTypes.InitOutput(
        doc="Schema for the output forced measurement catalogs.",
        name="{outputCoaddName}Coadd_forced_src_schema",
        storageClass="SourceCatalog",
    )
    exposure = pipeBase.connectionTypes.Input(
        doc="Input exposure to perform photometry on.",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=["band", "skymap", "tract", "patch"],
    )
    refCat = pipeBase.connectionTypes.Input(
        doc="Catalog of shapes and positions at which to force photometry.",
        name="{inputCoaddName}Coadd_ref",
        storageClass="SourceCatalog",
        dimensions=["skymap", "tract", "patch"],
    )
    refCatInBand = pipeBase.connectionTypes.Input(
        doc="Catalog of shapes and positions in the band having forced"
        "photometry done",
        name="{inputCoaddName}Coadd_meas",
        storageClass="SourceCatalog",
        dimensions=("band", "skymap", "tract", "patch"),
    )
    footprintCatInBand = pipeBase.connectionTypes.Input(
        doc="Catalog of footprints to attach to sources",
        name="{inputCoaddName}Coadd_deblendedFlux",
        storageClass="SourceCatalog",
        dimensions=("band", "skymap", "tract", "patch"),
    )
    scarletModels = pipeBase.connectionTypes.Input(
        doc="Multiband scarlet models produced by the deblender",
        name="{inputCoaddName}Coadd_scarletModelData",
        storageClass="ScarletModelData",
        dimensions=("tract", "patch", "skymap"),
    )
    refWcs = pipeBase.connectionTypes.Input(
        doc="Reference world coordinate system.",
        name="{inputCoaddName}Coadd_calexp.wcs",
        storageClass="Wcs",
        dimensions=["band", "skymap", "tract", "patch"],
    )  # used in place of a skymap wcs because of DM-28880
    measCat = pipeBase.connectionTypes.Output(
        doc="Output forced photometry catalog.",
        name="{outputCoaddName}Coadd_forced_src",
        storageClass="SourceCatalog",
        dimensions=["band", "skymap", "tract", "patch"],
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config.footprintDatasetName != "ScarletModelData":
            self.inputs.remove("scarletModels")
        if config.footprintDatasetName != "DeblendedFlux":
            self.inputs.remove("footprintCatInBand")


class ForcedPhotCoaddConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=ForcedPhotCoaddConnections
):
    measurement = lsst.pex.config.ConfigurableField(
        target=ForcedMeasurementTask, doc="subtask to do forced measurement"
    )
    coaddName = lsst.pex.config.Field(
        doc="coadd name: typically one of deep or goodSeeing",
        dtype=str,
        default="deep",
    )
    doApCorr = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Run subtask to apply aperture corrections",
    )
    applyApCorr = lsst.pex.config.ConfigurableField(
        target=ApplyApCorrTask, doc="Subtask to apply aperture corrections"
    )
    catalogCalculation = lsst.pex.config.ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog",
    )
    footprintDatasetName = lsst.pex.config.Field(
        doc="Dataset (without coadd prefix) that should be used to obtain "
        "(Heavy)Footprints for sources. Must have IDs that match those of the"
        "reference catalog. If None, Footprints will be generated by"
        "transforming the reference Footprints.",
        dtype=str,
        default="ScarletModelData",
        optional=True,
    )
    doConserveFlux = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Whether to use the deblender models as templates to re-distribute"
        "the flux from the exposure (True), or to perform measurements on the"
        "deblender model footprints. If "
        "footprintDatasetName != ScarletModelData then this field is ignored.",
    )
    doStripFootprints = lsst.pex.config.Field(
        dtype=bool,
        default=True,
        doc="Whether to strip footprints from the output catalog before "
        "saving to disk. "
        "This is usually done when using scarlet models to save disk space.",
    )
    hasFakes = lsst.pex.config.Field(
        dtype=bool,
        default=False,
        doc="Should be set to True if fake sources have been inserted into the"
        "input data.",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def setDefaults(self):
        # Docstring inherited.
        # Make catalogCalculation a no-op by default as no modelFlux is setup
        # by default in ForcedMeasurementTask
        super().setDefaults()

        self.catalogCalculation.plugins.names = []
        self.measurement.copyColumns["id"] = "id"
        self.measurement.copyColumns["parent"] = "parent"
        self.measurement.plugins.names |= ["base_Variance"]


class ForcedPhotCoaddTask(pipeBase.PipelineTask):
    """A pipeline task for performing forced measurement on coadd images.

    Parameters
    ----------
    refSchema : `lsst.afw.table.Schema`, optional
        The schema of the reference catalog, passed to the constructor of the
        references subtask. Optional, but must be specified if ``initInputs``
        is not; if both are specified, ``initInputs`` takes precedence.
    initInputs : `dict`
        Dictionary that can contain a key ``inputSchema`` containing the
        schema. If present will override the value of ``refSchema``.
    **kwds
        Keyword arguments are passed to the supertask constructor.
    """

    ConfigClass = ForcedPhotCoaddConfig
    _DefaultName = "forcedPhotCoadd"
    dataPrefix = "deepCoadd_"

    def __init__(self, refSchema=None, initInputs=None, **kwds):
        super().__init__(**kwds)

        if initInputs is not None:
            refSchema = initInputs["inputSchema"].schema

        if refSchema is None:
            raise ValueError("No reference schema provided.")
        self.makeSubtask("measurement", refSchema=refSchema)
        # It is necessary to get the schema internal to the forced measurement
        # task until such a time that the schema is not owned by the
        # measurement task, but is passed in by an external caller.
        if self.config.doApCorr:
            self.makeSubtask("applyApCorr", schema=self.measurement.schema)
        self.makeSubtask("catalogCalculation", schema=self.measurement.schema)
        self.outputSchema = lsst.afw.table.SourceCatalog(
            self.measurement.schema
        )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        refCatInBand = inputs.pop("refCatInBand")
        if self.config.footprintDatasetName == "ScarletModelData":
            footprintData = inputs.pop("scarletModels")
        elif self.config.footprintDatasetName == "DeblendedFlux":
            footprintData = inputs.pop("footprintCatIndBand")
        else:
            footprintData = None
        inputs["measCat"], inputs["exposureId"] = self.generateMeasCat(
            inputRefs.exposure.dataId,
            inputs["exposure"],
            inputs["refCat"],
            refCatInBand,
            inputs["refWcs"],
            footprintData,
        )
        outputs = self.run(**inputs)
        # Strip HeavyFootprints to save space on disk
        if (
            self.config.footprintDatasetName == "ScarletModelData"
            and self.config.doStripFootprints
        ):
            sources = outputs.measCat
            for source in sources[sources["parent"] != 0]:
                source.setFootprint(None)
        butlerQC.put(outputs, outputRefs)

    def generateMeasCat(
        self, dataId, exposure, refCat, refCatInBand, refWcs, footprintData
    ):
        """Generate a measurement catalog.

        Parameters
        ----------
        dataId : `lsst.daf.butler.DataCoordinate`
            Butler data ID for this image, with ``{tract, patch, band}`` keys.
        exposure : `lsst.afw.image.exposure.Exposure`
            Exposure to generate the catalog for.
        refCat : `lsst.afw.table.SourceCatalog`
            Catalog of shapes and positions at which to force photometry.
        refCatInBand : `lsst.afw.table.SourceCatalog`
            Catalog of shapes and position in the band forced photometry is
            currently being performed
        refWcs : `lsst.afw.image.SkyWcs`
            Reference world coordinate system.
        footprintData : `ScarletDataModel` or `lsst.afw.table.SourceCatalog`
            Either the scarlet data models or the deblended catalog containing
            footprints. If `footprintData` is `None` then the footprints
            contained in `refCatInBand` are used.

        Returns
        -------
        measCat : `lsst.afw.table.SourceCatalog`
            Catalog of forced sources to measure.
        expId : `int`
            Unique binary id associated with the input exposure

        Raises
        ------
        LookupError
            Raised if a footprint with a given source id was in the reference
            catalog but not in the reference catalog in band (meaning there was
            some sort of mismatch in the two input catalogs)
        """
        id_generator = self.config.idGenerator.apply(dataId)
        measCat = self.measurement.generateMeasCat(
            exposure,
            refCat,
            refWcs,
            idFactory=id_generator.make_table_id_factory(),
        )
        # attach footprints here as this can naturally live inside this method
        if self.config.footprintDatasetName == "ScarletModelData":
            # Load the scarlet models
            self._attachScarletFootprints(
                catalog=measCat,
                modelData=footprintData,
                exposure=exposure,
                band=dataId["band"],
            )
        else:
            if self.config.footprintDatasetName is None:
                footprintCat = refCatInBand
            else:
                footprintCat = footprintData
            for srcRecord in measCat:
                fpRecord = footprintCat.find(srcRecord.getId())
                if fpRecord is None:
                    raise LookupError(
                        "Cannot find Footprint for source {}; please check that {} "
                        "IDs are compatible with reference source IDs".format(
                            srcRecord.getId(), footprintCat
                        )
                    )
                srcRecord.setFootprint(fpRecord.getFootprint())
        return measCat, id_generator.catalog_id

    def run(self, measCat, exposure, refCat, refWcs, exposureId=None):
        """Perform forced measurement on a single exposure.

        Parameters
        ----------
        measCat : `lsst.afw.table.SourceCatalog`
            The measurement catalog, based on the sources listed in the
            reference catalog.
        exposure : `lsst.afw.image.Exposure`
            The measurement image upon which to perform forced detection.
        refCat : `lsst.afw.table.SourceCatalog`
            The reference catalog of sources to measure.
        refWcs : `lsst.afw.image.SkyWcs`
            The WCS for the references.
        exposureId : `int`
            Optional unique exposureId used for random seed in measurement
            task.

        Returns
        -------
        result : ~`lsst.pipe.base.Struct`
            Structure with fields:

            ``measCat``
                Catalog of forced measurement results
                (`lsst.afw.table.SourceCatalog`).
        """
        # We want to cache repeated PSF evaluations at the same point coming
        # from different measurement plugins.  We assume each algorithm tries
        # to evaluate the PSF twice, which is more than enough since many don't
        # evaluate it at all, and there's no *good* reason for any algorithm to
        # evaluate it more than once.
        exposure.psf.setCacheCapacity(
            2 * len(self.config.measurement.plugins.names)
        )
        self.measurement.run(
            measCat, exposure, refCat, refWcs, exposureId=exposureId
        )
        if self.config.doApCorr:
            self.applyApCorr.run(
                catalog=measCat,
                apCorrMap=exposure.getInfo().getApCorrMap(),
            )
        self.catalogCalculation.run(measCat)

        return pipeBase.Struct(measCat=measCat)

    def _attachScarletFootprints(self, catalog, modelData, exposure, band):
        """Attach scarlet models as HeavyFootprints"""
        if self.config.doConserveFlux:
            redistributeImage = exposure
        else:
            redistributeImage = None
        # Attach the footprints
        updateCatalogFootprints(
            modelData=modelData,
            catalog=catalog,
            band=band,
            imageForRedistribution=redistributeImage,
            removeScarletData=True,
            updateFluxColumns=False,
        )
