import utilities.logging as Logger
import integration.files as Files
import evaluation.general as GenEval

import evaluation.iv as IVEval

bib = Files.DataCollection()

Logger.setup(bib)
Logger.set_level(Logger.DEBUG)

Files.setup(bib, "test", "/home/dacap/Downloads")
bib.data.file_name = "OI-25c-09 2025-04-15 unbroken antenna full 1-formated.hdf5"

GenEval.setup(bib)
bib.params.volt_amp = (1.0, 1.0)

# Files.showFile(bib.data)

GenEval.loadMeasurements(bib)
GenEval.showLoadedMeasurements(bib)
GenEval.select_measurement(bib, 2)
GenEval.select_CurveSet(bib, 1)

IVEval.setup(bib)
IVEval.select_edge(bib, 1, "down")
IVEval.loadCurveSets(bib)
IVEval.filter_curve_sets(bib)
