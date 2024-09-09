"""
Tests BasePlotting.
"""

from utilities.baseplotting import BasePlotting

bp = BasePlotting()
bp.base["title"] = "test_baseplotting"
bp.base["data_folder"] = ".test/"
bp.base["sub_folder"] = ""

bp.loadData()
bp.x_key = "V_bias_up_mV"
bp.y_key = "T_axis_up_K"
bp.z_key = "dIdV_up_T"
bp.x_lim = [-1.3, 1.3]
bp.y_lim = [0.06, 1.4]
bp.z_lim = [0, 0.2]
bp.showMap()
bp.reshowMap()
bp.saveFigure()
