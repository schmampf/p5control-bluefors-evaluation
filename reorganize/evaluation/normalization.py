# region imports
# std

# third-party
import numpy as np

# local
from integration.files import DataCollection
import utilities.logging as Logger
import evaluation.iv as IVEval
import utilities.math as Math

# endregion

from matplotlib import pyplot as plt


def normalize(bib: DataCollection):
    Logger.print(
        Logger.INFO,
        msg=f"Normalization.normalize()",
    )
    bib.evaluation.persistent_sets["norm"] = IVEval.DataSet()
    norm = bib.evaluation.persistent_sets["norm"]

    voltage = np.copy(bib.evaluation.cached_sets["adwin"].curves["voltage-bin"])
    dIdV = np.copy(bib.evaluation.cached_sets["diffs"].curves["diff_conductance"])

    dIdV = dIdV[len(dIdV) // 2 :]
    voltage = voltage[len(voltage) // 2 :]
    idxDelta = 235  # np.argmax(dIdV)
    dx = 2 / idxDelta
    voltage = np.linspace(
        start=-dx * len(voltage),
        stop=dx * len(voltage),
        num=len(voltage) * 2,
    )
    norm.curves["voltage-bin"] = voltage


# def normedX(x, y):
#     y = Math.moving_average(y, 10)
#     y = Math.moving_average(y, 10)
#     y = Math.moving_average(y, 10)

#     # find all maxima in the curve
#     maxima = []
#     for i in range(1, len(y) - 1):
#         if y[i - 1] < y[i] > y[i + 1]:
#             maxima.append(i)

#     # get last maximum index
#     idxDelta = maxima[len(maxima) - 1]
#     dx = 2 / idxDelta
#     voltage = np.linspace(
#         start=-dx * len(x),
#         stop=dx * len(x),
#         num=len(x) * 2,
#     )
#     return voltage
