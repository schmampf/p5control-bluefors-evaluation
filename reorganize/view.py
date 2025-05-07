import utilities.logging as Logger
import integration.files as Files
import evaluation.general as GenEval
import evaluation.iv as IVEval

bib = Files.DataCollection()

Logger.setup(bib)
Logger.set_level(Logger.INFO)
Files.setup(bib, "Test", "/home/dacap/Downloads")
GenEval.setup(bib)
IVEval.setup(bib)

bib.data.file_name = "OI-25c-09 2025-04-15 unbroken antenna full 1-formated.hdf5"

Files.showFile(bib.data)
