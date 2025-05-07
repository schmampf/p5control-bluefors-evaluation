# region imports
# std

# third party
import numpy as np

# local
from integration.files import DataCollection
import utilities.logging as Logger
import evaluation.general as GenEval
import evaluation.iv as IVEval

# endregion


def load(bib: DataCollection, curve_index: int):
    """Load a single set of curves"""
    GenEval.select_CurveSet(bib, curve_index)
    IVEval.loadCurveSets(bib)


def eval(bib: DataCollection, filter_set: str):
    """Evaluate a single set of curves"""
    if not bib.iv_params.bins > 0:
        Logger.print(
            Logger.ERROR,
            msg="The number of bins must be greater than 0",
        )
        assert False

    IVEval.filter_curve_sets(bib, filter_set, bib.iv_params.bins)
    IVEval.process_curve_sets(bib)
    if filter_set == "adwin":
        IVEval.get_noise(bib, "adwin", ("time", "current"), "It")


def bulk_eval(bib: DataCollection):
    result = {}
    num_slices = bib.params.available_measurement_entries
    slice_index = np.linspace(0, num_slices - 1, num_slices, dtype=int)
    init: bool = False

    VVI = np.empty((num_slices, 0), dtype=float)

    Logger.suppress_print = True
    for i in slice_index:
        Logger.print(
            Logger.INFO,
            msg=f"Evaluating Set: {i+1}/{num_slices} ({i/num_slices:.0%}) (id: {bib.params.selected_dataset})",
            force=True,
            updating=True,
        )

        load(bib, i)
        eval(bib, "adwin")
        eval(bib, "bluefors")

        iv_x = bib.evaluation.cached_sets["adwin"].curves["voltage-bin"]
        iv_z = bib.evaluation.cached_sets["adwin"].curves["current-voltage"]

        if not init:
            VVI = np.empty((num_slices, iv_x.size), dtype=float)
            init = True

        VVI[i, :] = np.gradient(np.array(iv_z, dtype=float))
    Logger.suppress_print = False

    iv_y_labels = [
        GenEval.MeasurementHeader.parse_number(l)[0]
        for l in bib.params.available_measurement_entries_labels
    ]
    result["VVI"] = IVEval.Map(
        x_axis=IVEval.Axis(
            name="Voltage (V)",
            values=iv_x,
        ),
        y_axis=IVEval.Axis(
            name="Voltage (V)",
            values=np.array(iv_y_labels),
        ),
        z_axis=IVEval.Axis(
            name="Current (I)",
            values=iv_z,
        ),
        values=VVI,
    )

    bib.result.maps = result
