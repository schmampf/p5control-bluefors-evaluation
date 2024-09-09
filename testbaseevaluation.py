"""
Tests BaseEvaluation.
"""

from utilities.baseevaluation import BaseEvaluation

be = BaseEvaluation("base eva")
be.base["title"] = "test_baseevaluation"
be.base["data_folder"] = ".test/"
be.base["figure_folder"] = ".test/"
be.base["sub_folder"] = ""

be.file["folder"] = "24 07 OI-24d-10/unbroken/24 08 13 temperature study/"
be.file["name"] = "OI-24d-10 24-08-13 temperature study 4.hdf5"

be.showAmplifications()
be.setAmplifications(1000, 10000)
be.showMeasurements()
be.setMeasurement("temperatures")
be.showKeys()
be.setKeys([7, -2, 1e-6, "no_power"])
be.setV(1.4e-3, voltage_bins=700)
be.getMaps()
be.setT(0, 2, 2000)
be.getMapsT()

be.saveData()
be.loadData()
be.showData()

be.base["title"] = "test_baseplotting"
be.saveData()
