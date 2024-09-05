"""
Tests Baseclass.
"""

from utilities.baseclass import BaseClass

bc = BaseClass("base")
bc.base["title"] = "test_baseclass"
bc.base["data_folder"] = ".test/"
bc.base["sub_folder"] = ""
bc.saveData()
bc.loadData()
bc.showData()
