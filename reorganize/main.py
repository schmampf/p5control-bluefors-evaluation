import numpy as np

import integration.files as Files
import evaluation.general as GenEval

bib = Files.DataCollection()
Files.setup(bib, "test", "/home/dacap/Downloads")
bib.data.title = "Test Tile"
bib.data.file_name = "23_11_15_PR22e9_4.3G_1.hdf5"
GenEval.setup(bib)

GenEval.loadMeasurements(bib)

# Files.showFile(bib.data)
