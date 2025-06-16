# region imports
# std

# third party
import numpy as np
from scipy.interpolate import interp1d

# local
from integration.files import DataCollection
import utilities.logging as Logger
import evaluation.general as GenEval
import evaluation.iv as IVEval
import evaluation.normalization as Norm
import utilities.math as Math
import algorithms.binning as Binning

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

    v_norm = []

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

        t_b = bib.evaluation.cached_sets["adwin"].curves["time-bin"]
        v_b = bib.evaluation.cached_sets["adwin"].curves["voltage-bin"]
        i_b = bib.evaluation.cached_sets["adwin"].curves["current-bin"]

        iv = bib.evaluation.cached_sets["adwin"].curves["current-voltage"]
        vi = bib.evaluation.cached_sets["adwin"].curves["voltage-current"]
        d_res = bib.evaluation.cached_sets["diffs"].curves["diff_resistance"]
        d_cond = bib.evaluation.cached_sets["diffs"].curves["diff_conductance"]

        if bib.params.evalTemperature:
            eval(bib, "bluefors")
            temp_b = bib.evaluation.cached_sets["bluefors"].curves["temperature"]
            m_temp = bib.evaluation.cached_sets["diffs"].curves["mean_temp"]
            TX[i, 0] = np.array(m_temp, dtype=float)

        if i == 0:
            Norm.normalize(bib)

        VXI[i, :] = np.array(iv, dtype=float)
        IXV[i, :] = np.array(vi, dtype=float)
        dIXR[i, :] = np.array(d_res, dtype=float)
        dVXC[i, :] = np.array(d_cond, dtype=float)

    Logger.suppress_print = False
    Logger.print(
        Logger.INFO,
        Logger.START,
        msg="\nFinished evaluating all sets",
    )

    y_labels = [
        GenEval.MeasurementHeader.parse_number(l)[0]
        for l in bib.params.available_measurement_entries_labels
    ]

    if bib.params.linearizeYAxis:
        Logger.print(
            Logger.INFO,
            msg="Linearizing Y-axis values from dBm to V",
        )
        y_labels = [Math.dBmToV(float(label)) for label in y_labels]
        y_labels = sorted(y_labels)
        ybins = np.linspace(
            min(y_labels),
            max(y_labels),
            bib.iv_params.bins,
        )

        interpolator = interp1d(y_labels, VXI, axis=0, kind="linear")
        VXI = interpolator(ybins)
        interpolator = interp1d(y_labels, IXV, axis=0, kind="linear")
        IXV = interpolator(ybins)
        interpolator = interp1d(y_labels, dVXC, axis=0, kind="linear")
        dVXC = interpolator(ybins)
        interpolator = interp1d(y_labels, dIXR, axis=0, kind="linear")
        dIXR = interpolator(ybins)

    if bib.params.smoothData[0]:
        Logger.print(
            Logger.INFO,
            msg="Smoothing data with moving average",
        )

        def smoothMatrix(matrix):
            """Smooth a matrix with a moving average."""
            smoothed = np.empty_like(matrix)
            for i in range(matrix.shape[0]):  # for each curve
                for j in range(1, bib.params.smoothData[1]):  # how many times to smooth
                    smoothed[i, :] = Math.moving_average(matrix[i, :], 10)
            return smoothed

        VXI = smoothMatrix(VXI)
        IXV = smoothMatrix(IXV)
        dIXR = smoothMatrix(dIXR)
        dVXC = smoothMatrix(dVXC)

    norm = bib.params.normalizeXAxis
    v_norm = bib.evaluation.persistent_sets["norm"].curves["voltage-bin"]

    result["VXI"] = IVEval.Map(
        x_axis=IVEval.Axis(
            name=r"Voltage ($V$)",
            values=np.array(v_norm) if norm else np.array(v_b),
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
            values=np.array(v_norm) if norm else np.array(v_b),
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
