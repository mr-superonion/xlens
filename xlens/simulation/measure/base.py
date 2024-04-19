#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 20082014 LSST Corpoalphan.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import os

import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms import ScaleVarianceTask, SourceDetectionTask
from lsst.meas.base import CatalogCalculationTask, SingleFrameMeasurementTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.utils import getPackageDir

from ..simulator.loader import MakeDMExposure


class DMMeasurementConfig(pexConfig.Config):
    "configuration"
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect sources",
    )
    scaleVariance = pexConfig.ConfigurableField(
        target=ScaleVarianceTask,
        doc="Variance rescaling",
    )
    deblend = pexConfig.ConfigurableField(
        target=SourceDeblendTask,
        doc="Deblending",
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Measure sources",
    )
    catalogCalculation = pexConfig.ConfigurableField(
        target=CatalogCalculationTask,
        doc="Subtask to run catalogCalculation plugins on catalog",
    )

    def setDefaults(self):
        super().setDefaults()
        self.detection.isotropicGrow = True
        self.detection.reEstimateBackground = False
        self.detection.thresholdValue = 5.0

        self.deblend.propagateAllPeaks = True
        self.deblend.maxFootprintArea = 500 * 500
        self.deblend.maxFootprintSize = 600

        # self.measurement.load(os.path.join(getPackageDir("obs_subaru"), "config", "hsm.py"))
        # self.load(os.path.join(getPackageDir("obs_subaru"), "config", "cmodel.py"))


class DMMeasurementTask(pipeBase.PipelineTask):
    ConfigClass = DMMeasurementConfig
    _DefaultName = "DMMeasurementTask"

    def __init__(self, do_deblend=True, do_scale_variance=True, **kwargs):
        super().__init__(**kwargs)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.do_deblend = do_deblend
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask("detection", schema=self.schema)
        self.do_scale_variance = do_scale_variance
        if do_deblend:
            self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask(
            "measurement", schema=self.schema, algMetadata=self.algMetadata
        )
        self.makeSubtask("catalogCalculation", schema=self.schema)
        if do_scale_variance:
            self.makeSubtask("scaleVariance")

    def measure_exposure(self, exposure):
        # Read galaxy exposure
        if not exposure.hasPsf():
            self.log.info("exposure doesnot have PSF")
            return None

        table = afwTable.SourceTable.make(self.schema)
        sources = afwTable.SourceCatalog(table)
        table.setMetadata(self.algMetadata)
        detRes = self.detection.run(
            table=table,
            exposure=exposure,
            doSmooth=True,
        )
        sources = detRes.sources
        print("number of detections: %d" % len(sources))
        if self.do_scale_variance:
            varScale = self.scaleVariance.run(exposure.maskedImage)
            exposure.getMetadata().add("variance_scale", varScale)
        if self.do_scale_variance:
            # do deblending
            self.deblend.run(exposure=exposure, sources=sources)
        # do measurement
        print("Start DMMeasurement")
        self.measurement.run(measCat=sources, exposure=exposure)
        # measurement on the catalog level
        print("Start Catalog Calculation")
        self.catalogCalculation.run(sources)
        return sources

    def run(self, exposure):
        return self.measure_exposure(exposure)


class ProcessSimDM(MakeDMExposure):
    def __init__(self, config_name):
        super().__init__(config_name)
        self.dm_task = DMMeasurementTask()
        self.output_dir = self.cat_dm_dir
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def run(self, file_name):
        out_name = os.path.join(self.output_dir, file_name.split("/")[-1])
        out_name = out_name.replace("image-", "src-").replace(
            "_xxx", "_%s" % self.bands
        )
        if os.path.isfile(out_name):
            print("Already has the output file")
            return
        exposure = self.generate_exposure(file_name)
        sources = self.dm_task.measure_exposure(exposure)
        sources.writeFits(out_name)
        return
