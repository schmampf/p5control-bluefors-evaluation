import evaluation.iv as IVEval
import evaluation.general as GenEval
import utilities.logging as Logger
import integration.files as Files
from integration.files import DataCollection
import algorithms.binning as Binning

import numpy as np


def eval(bib: DataCollection, curve_index: int, filter_set: str):
    """Evaluate a single set of curves"""
    GenEval.select_CurveSet(bib, curve_index)
    IVEval.loadCurveSets(bib)
    # print(bib.evaluation.cached_sets["adwin"].curves.keys())
    IVEval.filter_curve_sets(bib, filter_set, 1000)
    # print(bib.evaluation.cached_sets["adwin"].curves.keys())
    IVEval.process_curve_sets(bib)
    # print(bib.evaluation.cached_sets["adwin"].curves.keys())
    if filter_set == "adwin":
        IVEval.get_noise(bib, "adwin", ("time", "current"), "It")


def bulk_eval(bib: DataCollection) -> dict[str, np.ndarray]:
    result = {}
    num_slices = bib.params.available_measurement_entries
    slice_index = np.linspace(0, num_slices - 1, num_slices, dtype=int)
    init: bool = False

    result["VVI"] = np.empty((num_slices, 0), dtype=float)

    Logger.suppress_print = True
    for i in slice_index:
        print(f"\rEvaluating {i}/{num_slices}", end="")

        eval(bib, i, "adwin")
        eval(bib, i, "bluefors")

        iv_y = bib.evaluation.cached_sets["adwin"].curves["current-voltage"]
        iv_x = bib.evaluation.cached_sets["adwin"].curves["voltage-bin"]

        if not init:
            result["VVI"] = np.empty((num_slices, iv_x.size), dtype=float)
            init = True

        result["VVI"][i, :] = np.array(iv_y, dtype=float)
    Logger.suppress_print = False

    return result
