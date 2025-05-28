# region imports
# std

# third-party
import numpy as np

# local
from integration.files import DataCollection
import utilities.logging as Logger
import evaluation.iv as IVEval

# endregion


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
    idxDelta = np.argmax(dIdV)
    dx = 2 / idxDelta
    voltage = np.linspace(
        start=-dx * len(voltage),
        stop=dx * len(voltage),
        num=len(voltage),
    )
    norm.curves["voltage-bin"] = voltage
