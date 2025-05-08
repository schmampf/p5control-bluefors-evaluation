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

    VXI = np.empty((num_slices, bib.iv_params.bins), dtype=float)
    IXV = np.empty((num_slices, bib.iv_params.bins), dtype=float)
    dIXR = np.empty((num_slices, bib.iv_params.bins), dtype=float)
    dVXC = np.empty((num_slices, bib.iv_params.bins), dtype=float)

    TX = np.empty((num_slices, 1), dtype=float)

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

        t_b = bib.evaluation.cached_sets["adwin"].curves["time-bin"]
        v_b = bib.evaluation.cached_sets["adwin"].curves["voltage-bin"]
        i_b = bib.evaluation.cached_sets["adwin"].curves["current-bin"]
        temp_b = bib.evaluation.cached_sets["bluefors"].curves["temperature"]

        iv = bib.evaluation.cached_sets["adwin"].curves["current-voltage"]
        vi = bib.evaluation.cached_sets["adwin"].curves["voltage-current"]
        d_res = bib.evaluation.cached_sets["diffs"].curves["diff_resistance"]
        d_cond = bib.evaluation.cached_sets["diffs"].curves["diff_conductance"]
        m_temp = bib.evaluation.cached_sets["diffs"].curves["mean_temp"]

        VXI[i, :] = np.array(iv, dtype=float)
        IXV[i, :] = np.array(vi, dtype=float)
        dIXR[i, :] = np.array(d_res, dtype=float)
        dVXC[i, :] = np.array(d_cond, dtype=float)

        TX[i, 0] = np.array(m_temp, dtype=float)
    Logger.suppress_print = False

    y_labels = [
        GenEval.MeasurementHeader.parse_number(l)[0]
        for l in bib.params.available_measurement_entries_labels
    ]

    result["VXI"] = IVEval.Map(
        x_axis=IVEval.Axis(
            name=r"Voltage ($V$)",
            values=v_b,
        ),
        y_axis=IVEval.Axis(
            name="X",
            values=np.array(y_labels),
        ),
        z_axis=IVEval.Axis(name=r"Current ($I$)"),
        values=VXI,
    )
    result["IXV"] = IVEval.Map(
        x_axis=IVEval.Axis(
            name=r"Current ($I$)",
            values=i_b,
        ),
        y_axis=IVEval.Axis(
            name="X",
            values=np.array(y_labels),
        ),
        z_axis=IVEval.Axis(name=r"Voltage ($V$)"),
        values=IXV,
    )
    result["dIXR"] = IVEval.Map(
        x_axis=IVEval.Axis(
            name=r"Current ($I$)",
            values=i_b,
        ),
        y_axis=IVEval.Axis(
            name="X",
            values=np.array(y_labels),
        ),
        z_axis=IVEval.Axis(name=r"d$V$/d$I$ ($O$)"),
        values=dIXR,
    )
    result["dVXC"] = IVEval.Map(
        x_axis=IVEval.Axis(
            name=r"Voltage (V)",
            values=v_b,
        ),
        y_axis=IVEval.Axis(
            name="X",
            values=np.array(y_labels),
        ),
        z_axis=IVEval.Axis(name=r"d$I$/d$V$ ($G_0$)"),
        values=dVXC,
    )

    result["TX"] = IVEval.Curve(
        dependent_ax=IVEval.Axis(
            name="X",
            values=np.array(y_labels),
        ),
        values=TX,
    )
    bib.result.maps = result
