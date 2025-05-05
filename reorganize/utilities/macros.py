import evaluation.iv as IVEval
import evaluation.general as GenEval
import utilities.logging as Logger
import integration.files as Files
from integration.files import DataCollection
import algorithms.binning as Binning


def macro_eval(bib: DataCollection, curve_index: int, filter_set: str):
    GenEval.select_CurveSet(bib, curve_index)
    IVEval.loadCurveSets(bib)
    IVEval.filter_curve_sets(bib, filter_set, 43)
    IVEval.process_curve_sets(bib)
    IVEval.get_noise(bib, "adwin", ("time", "current"), "IV")
