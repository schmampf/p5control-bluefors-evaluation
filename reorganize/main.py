import utilities.logging as Logger
import integration.files as Files
import evaluation.general as GenEval

bib = Files.DataCollection()

Logger.setup(bib)
Logger.set_level(Logger.INFO)

Files.setup(bib, "test", "/home/dacap/Downloads")
bib.data.file_name = "OI-25c-09 2025-04-15 unbroken 0.copy.hdf5"

GenEval.setup(bib)
# Files.showFile(bib.data)
GenEval.loadMeasurements(bib)
GenEval.showLoadedMeasurements(bib)
GenEval.select_measurement(bib, 10)
