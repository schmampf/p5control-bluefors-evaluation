import numpy as np

import integration.files as Files
import evaluation.general as GenEval

bib = Files.DataCollection()
Files.setup(bib, "test", "C:\\Users\\capal\\Downloads")
bib.data.title = "Test Tile"
bib.data.file_name = "23_11_10_PR22e9_1.2G_1.hdf5"
GenEval.setup(bib)

# mh = GenEval.MeasurementHeader.new("gate_voltages vna_nanGHz_nanV magnet_-0.000001mT")

# print(mh)
# GenEval.loadMeasurements(bib)

# Files.showFile(bib.data)

# GenEval.loadMeasurements(bib)

print(GenEval.MeasurementHeader.from_string("vna_nanGHz_nanV magnet_-0.000001mT"))
