from lsst.pipe.base import (
    Struct,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
)
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field, ConfigurableField
from lsst.meas.algorithms import SetPrimaryFlagsTask
from lsst.meas.astrom import DirectMatchTask
from lsst.meas.base import (
    SingleFrameMeasurementTask,
    CatalogCalculationTask,
    SkyMapIdGeneratorConfig,
)
from lsst.meas.extensions.scarlet.io import updateCatalogFootprints
import lsst.afw.table as afwTable
from lsst.daf.base import PropertyList
from lsst.skymap import BaseSkyMap


class MeasureMergedCoaddSourcesConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "deep",
        "deblendedCatalog": "deblendedFlux",
    },
):
    inputSchema = cT.InitInput(
        doc=(
            "Input schema for measure merged task produced by"
            "a deblender or detection task"
        ),
        name="{inputCoaddName}Coadd_deblendedFlux_schema",
        storageClass="SourceCatalog",
    )
    outputSchema = cT.InitOutput(
        doc="Output schema after all new fields are added by task",
        name="{inputCoaddName}Coadd_meas_schema",
        storageClass="SourceCatalog",
    )
    refCat = cT.Input(
        doc="Reference catalog used to match measured sources against known"
        "sources",
        name="ref_cat",
        storageClass="SimpleCatalog",
        dimensions=(
            "tract",
            "patch",
            "skymap",
        ),
        deferLoad=True,
        multiple=True,
        minimum=0,
    )
    exposure = cT.Input(
        doc="Input coadd image",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap"),
    )
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    scarletCatalog = cT.Input(
        doc="Catalogs produced by multiband deblending",
        name="{inputCoaddName}Coadd_deblendedCatalog",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )
    scarletModels = cT.Input(
        doc="Multiband scarlet models produced by the deblender",
        name="{inputCoaddName}Coadd_scarletModelData",
        storageClass="ScarletModelData",
        dimensions=("tract", "patch", "skymap"),
    )
    outputSources = cT.Output(
        doc=(
            "Source catalog containing all the measurement information"
            "generated in this task"
        ),
        name="{outputCoaddName}Coadd_meas",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="SourceCatalog",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.doAddFootprints:
            del self.scarletModels


class MeasureMergedCoaddSourcesConfig(
    PipelineTaskConfig,
    pipelineConnections=MeasureMergedCoaddSourcesConnections,
):
    """Configuration parameters for the MeasureMergedCoaddSourcesTask"""

    doAddFootprints = Field(
        dtype=bool,
        default=True,
        doc="Whether to add footprints to the input catalog from scarlet"
        "models. This should be true whenever using the multi-band deblender,"
        "otherwise this should be False.",
    )
    doConserveFlux = Field(
        dtype=bool,
        default=True,
        doc="Whether to use the deblender models as templates to re-distribute"
        "the flux from the 'exposure' (True), or to perform measurements on"
        "the deblender model footprints.",
    )
    doStripFootprints = Field(
        dtype=bool,
        default=True,
        doc="Whether to strip footprints from the output catalog before "
        "saving to disk."
        "This is usually done when using scarlet models to save disk space.",
    )
    measurement = ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Source measurement",
    )
    setPrimaryFlags = ConfigurableField(
        target=SetPrimaryFlagsTask,
        doc="Set flags for primary tract/patch",
    )
    doMatchSources = Field(
        dtype=bool,
        default=False,
        doc="Match sources to reference catalog?",
        deprecated="Reference matching will be removed after v29.",
    )
    match = ConfigurableField(
        target=DirectMatchTask,
        doc="Matching to reference catalog",
        deprecated="Reference matching will be removed after v29.",
    )
    doWriteMatchesDenormalized = Field(
        dtype=bool,
        default=False,
        doc=(
            "Write reference matches in denormalized format? "
            "This format uses more disk space, but more convenient to read."
        ),
        deprecated="Reference matching will be removed after v29.",
    )
    psfCache = Field(dtype=int, default=100, doc="Size of psfCache")
    checkUnitsParseStrict = Field(
        doc="Strictness of Astropy unit compatibility check, can be 'raise',"
        "'warn' or 'silent'",
        dtype=str,
        default="raise",
    )
    catalogCalculation = ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        self.measurement.plugins.names |= [
            "base_InputCount",
            "base_Variance",
            "base_LocalPhotoCalib",
            "base_LocalWcs",
        ]

    def validate(self):
        super().validate()


class MeasureMergedCoaddSourcesTask(PipelineTask):
    """Deblend sources from main catalog in each coadd seperately and measure.

    Use peaks and footprints from a master catalog to perform deblending and
    measurement in each coadd.

    Given a master input catalog of sources (peaks and footprints) or deblender
    outputs(including a HeavyFootprint in each band), measure each source on
    the coadd. Repeating this procedure with the same master catalog across
    multiple coadds will generate a consistent set of child sources.

    The deblender retains all peaks and deblends any missing peaks (dropouts in
    that band) as PSFs. Source properties are measured and the @c is-primary
    flag (indicating sources with no children) is set. Visit flags are
    propagated to the coadd sources.

    Optionally, we can match the coadd sources to an external reference
    catalog.

    After MeasureMergedCoaddSourcesTask has been run on multiple coadds, we
    have a set of per-band catalogs. The next stage in the multi-band
    processing procedure will merge these measurements into a suitable catalog
    for driving forced photometry.

    Parameters
    ----------
    schema : ``lsst.afw.table.Schema`, optional
        The schema of the merged detection catalog used as input to this one.
    peakSchema : ``lsst.afw.table.Schema`, optional
        The schema of the PeakRecords in the Footprints in the merged detection
        catalog.
    refObjLoader : `lsst.meas.algorithms.ReferenceObjectLoader`, optional
        An instance of ReferenceObjectLoader that supplies an external
        reference catalog. May be None if the loader can be constructed from
        the butler argument or all steps requiring a reference catalog are
        disabled.
    initInputs : `dict`, optional
        Dictionary that can contain a key ``inputSchema`` containing the
        input schema. If present will override the value of ``schema``.
    **kwargs
        Additional keyword arguments.
    """

    _DefaultName = "measureCoaddSources"
    ConfigClass = MeasureMergedCoaddSourcesConfig

    def __init__(
        self,
        schema=None,
        peakSchema=None,
        refObjLoader=None,
        initInputs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if initInputs is not None:
            schema = initInputs["inputSchema"].schema
        if schema is None:
            raise ValueError("Schema must be defined.")
        self.schemaMapper = afwTable.SchemaMapper(schema)
        self.schemaMapper.addMinimalSchema(schema)
        self.schema = self.schemaMapper.getOutputSchema()
        afwTable.CoordKey.addErrorFields(self.schema)
        self.algMetadata = PropertyList()
        self.makeSubtask(
            "measurement", schema=self.schema, algMetadata=self.algMetadata
        )
        self.makeSubtask("setPrimaryFlags", schema=self.schema)
        self.schema.checkUnits(parse_strict=self.config.checkUnitsParseStrict)
        self.makeSubtask("catalogCalculation", schema=self.schema)
        self.outputSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        # Set psfcache
        # move this to run after gen2 deprecation
        inputs["exposure"].getPsf().setCacheCapacity(self.config.psfCache)

        # Get unique integer ID for IdFactory and RNG seeds; only the latter
        # should really be used as the IDs all come from the input catalog.
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        inputs["exposureId"] = idGenerator.catalog_id

        # Transform inputCatalog
        table = afwTable.SourceTable.make(
            self.schema, idGenerator.make_table_id_factory()
        )
        sources = afwTable.SourceCatalog(table)
        # Load the correct input catalog
        inputCatalog = inputs.pop("scarletCatalog")
        catalogRef = inputRefs.scarletCatalog
        sources.extend(inputCatalog, self.schemaMapper)

        # Add the HeavyFootprints to the deblended sources
        if self.config.doAddFootprints:
            modelData = inputs.pop("scarletModels")
            if self.config.doConserveFlux:
                imageForRedistribution = inputs["exposure"]
            else:
                imageForRedistribution = None
            updateCatalogFootprints(
                modelData=modelData,
                catalog=sources,
                band=inputRefs.exposure.dataId["band"],
                imageForRedistribution=imageForRedistribution,
                removeScarletData=True,
                updateFluxColumns=True,
            )
        table = sources.getTable()
        table.setMetadata(
            self.algMetadata
        )  # Capture algorithm metadata to write out to the source catalog.
        inputs["sources"] = sources

        skyMap = inputs.pop("skyMap")
        tractNumber = catalogRef.dataId["tract"]
        tractInfo = skyMap[tractNumber]
        patchInfo = tractInfo.getPatchInfo(catalogRef.dataId["patch"])
        skyInfo = Struct(
            skyMap=skyMap,
            tractInfo=tractInfo,
            patchInfo=patchInfo,
            wcs=tractInfo.getWcs(),
            bbox=patchInfo.getOuterBBox(),
        )
        inputs["skyInfo"] = skyInfo

        outputs = self.run(**inputs)
        # Strip HeavyFootprints to save space on disk
        sources = outputs.outputSources
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        exposure,
        sources,
        skyInfo,
        exposureId,
        ccdInputs=None,
        sourceTableHandleDict=None,
        finalizedSourceTableHandleDict=None,
        **kwargs,
    ):
        """Run measurement algorithms on the input exposure, and optionally
        populate the resulting catalog with extra information.

        Parameters
        ----------
        exposure : `lsst.afw.exposure.Exposure`
            The input exposure on which measurements are to be performed.
        sources :  `lsst.afw.table.SourceCatalog`
            A catalog built from the results of merged detections, or deblender
            outputs.
        skyInfo : `lsst.pipe.base.Struct`
            A struct containing information about the position of the input
            exposure within a `SkyMap`, the `SkyMap`, its `Wcs`, and its
            bounding box.
        exposureId : `int` or `bytes`
            Packed unique number or bytes unique to the input exposure.
        ccdInputs : `lsst.afw.table.ExposureCatalog`, optional
            Catalog containing information on the individual visits which went
            into making the coadd.
        sourceTableHandleDict : `dict` [`int`,
            `lsst.daf.butler.DeferredDatasetHandle`], optional
            Dict for sourceTable_visit handles (key is visit) for propagating
            flags. These tables are derived from the ``CalibrateTask`` sources,
            and contain astrometry and photometry flags, and optionally PSF
            flags.
        finalizedSourceTableHandleDict : `dict` [`int`,
            `lsst.daf.butler.DeferredDatasetHandle`], optional
            Dict for finalized_src_table handles (key is visit) for propagating
            flags. These tables are derived from ``FinalizeCalibrationTask``
            and contain PSF flags from the finalized PSF estimation.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results of running measurement task. Will contain the catalog in the
            sources attribute. Optionally will have results of matching to a
            reference catalog in the matchResults attribute, and denormalized
            matches in the denormMatches attribute.
        """
        self.measurement.run(sources, exposure, exposureId=exposureId)
        if not sources.isContiguous():
            sources = sources.copy(deep=True)
        self.catalogCalculation.run(sources)
        self.setPrimaryFlags.run(
            sources,
            skyMap=skyInfo.skyMap,
            tractInfo=skyInfo.tractInfo,
            patchInfo=skyInfo.patchInfo,
        )
        results = Struct(outputSources=sources)
        return results
