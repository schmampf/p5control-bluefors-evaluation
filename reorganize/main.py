import utilities.logging as Logger
import integration.files as Files
import evaluation.general as GenEval

bib = Files.DataCollection()

Logger.setup(bib)
Logger.set_level(Logger.INFO)

Files.setup(bib, "test", "C:\\Users\\capal\\Downloads")
bib.data.title = "Test Tile"
bib.data.file_name = "23_11_10_PR22e9_1.2G_1.hdf5"

GenEval.setup(bib)
GenEval.loadMeasurements(bib)
GenEval.showLoadedMeasurements(bib)
GenEval.select_measurement(bib, 10)

# Files.showData(bib)
