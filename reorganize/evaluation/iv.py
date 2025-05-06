# region imports
# std
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

# third-party
import numpy as np
import h5py
from scipy import constants

# local
import integration.files as Files
from integration.files import DataCollection
import utilities.logging as Logger
import algorithms.binning as Binning

# endregion


# region dataclasses
@dataclass
class Axis:
    name: str = field(default="")
    values: np.ndarray = field(default_factory=lambda: np.array([]))

    @staticmethod
    def new(
        name: str,
        absolutes: float = np.nan,
        range: tuple[float, float] = (np.nan, np.nan),
        bins: int = 0,
    ):
        if absolutes is not np.nan:
            return Axis(name, np.linspace(-absolutes, absolutes, bins + 1))

        if range != (np.nan, np.nan):
            min, max = range
            return Axis(name, np.linspace(min, max, bins + 1))

        raise ValueError("Axis must be initialized with either absolutes or range.")

    @property
    def min(self):
        return self.values[0]

    @property
    def max(self):
        return self.values[-1]

    @property
    def bins(self):
        if len(self.values) == 0:
            return 0
        return len(self.values) - 1


@dataclass
class Curve:
    name: str = field(default="")
    dependent_ax: Axis = field(default_factory=Axis)
    values: np.ndarray = field(
        default_factory=lambda: np.full((0, 0), np.nan, dtype=np.float64)
    )

    def __post_init__(self):
        if np.shape(self.values) == (0, 0):
            self.values = np.full((self.dependent_ax.bins,), np.nan, dtype=np.float64)

    def __setattr__(self, name: str, value: np.ndarray) -> None:
        if name == "values":
            if len(value) != self.dependent_ax.bins:
                raise ValueError(
                    f"Values must have the same length as the number of bins ({self.dependent_ax.bins})."
                )
        super().__setattr__(name, value)


@dataclass
class DataSet:
    name: str = field(default="")
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    voltage_offsets: tuple[float, float] = field(default_factory=tuple)

    def check_curves(self):
        num_points = None
        for key, curve in self.curves.items():
            if num_points is None:
                num_points = len(curve)
            else:
                assert (
                    len(curve) == num_points
                ), f"The number of points in the curves is not equal: {num_points} != {len(curve)}"

    def filter_trigger(self, params: "Parameters") -> "DataSet | None":
        if "trigger" not in self.curves:
            Logger.print(
                Logger.ERROR,
                msg="No trigger curve found in the dataset.",
            )
            return None

        new_set = DataSet()
        new_set.name = self.name
        new_set.voltage_offsets = self.voltage_offsets

        trigger_index = params.edge_num * trigger_num(params.edge_dir) + (
            params.edge_num - 1 if params.edge_dir == "up" else 0
        )
        trigger_dat = self.curves["trigger"]

        for key, curve in self.curves.items():
            new_set.curves[key] = curve[trigger_dat == trigger_index].copy()

        return new_set

    def copy(self) -> "DataSet":
        new_set = DataSet()
        new_set.name = self.name
        new_set.voltage_offsets = self.voltage_offsets
        new_set.curves = {key: curve.copy() for key, curve in self.curves.items()}
        return new_set


@dataclass
class Map:
    name: str = field(default="")
    x_axis: Axis = field(default_factory=Axis)
    y_axis: Axis = field(default_factory=Axis)
    values: np.ndarray = field(
        default_factory=lambda: np.full((0, 0), np.nan, dtype=np.float64)
    )

    def __post_init__(self):
        self.values = np.full(
            (self.x_axis.bins, self.y_axis.bins), np.nan, dtype=np.float64
        )


@dataclass
class Result:
    raw_sets: dict[str, DataSet] = field(
        default_factory=dict
    )  # raw data directly after loading and minimal preprocessing
    cached_sets: dict[str, DataSet] = field(
        default_factory=dict
    )  # cached data of the last "operation" <-- main container after loading
    filtered_sets: dict[str, DataSet] = field(
        default_factory=dict
    )  # data after filtering
    processed_sets: dict[str, DataSet] = field(
        default_factory=dict
    )  # data after processing
    # temperatures: Axis = field(default_factory=Axis)  # one per dataset
    # time_starts: Axis = field(default_factory=Axis)  # one per dataset
    # time_stops: Axis = field(default_factory=Axis)  # one per dataset
    # temp_current: Map = field(default_factory=Map)
    # temp_voltage: Map = field(default_factory=Map)
    # diff_resistance: Map = field(default_factory=Map)
    # diff_conductance: Map = field(default_factory=Map)


@dataclass
class Configuration:
    eva_current: bool = True
    eva_voltage: bool = True
    eva_temperature: bool = True
    eva_even_spaced: bool = False


@dataclass
class Parameters:
    edge_num: int = 0
    edge_dir: str = field(default="nan")


# endregion


# region helper
def trigger_dir(trigger: int) -> str:
    # 0 = nan
    # 1 = up
    # 2 = down
    # ...
    if trigger == 0:
        return "nan"
    if trigger % 2 == 1:
        return "up"  # odd triggers
    else:
        return "down"  # even triggers


def trigger_num(trigger: str) -> int:
    if trigger == "up":
        return 1
    if trigger == "down":
        return 2
    else:
        raise ValueError(
            f"Invalid trigger direction: {trigger}. Must be 'up' or 'down'."
        )


# endregion


def setup(bib: DataCollection):
    bib.iv_config = Configuration()
    bib.iv_params = Parameters()
    bib.evaluation = Result()

    Logger.print(Logger.INFO, Logger.START, "IVEval.setup()")


def select_edge(
    collection: DataCollection,
    num: int,
    dir: str,
):
    Logger.print(
        Logger.INFO,
        Logger.START,
        f"IVEval.select_edge(num={num},dir={dir})",
    )
    assert dir in ["up", "down"], Logger.print(
        Logger.ERROR,
        msg=f"Invalid edge direction: {dir}. Must be 'up' or 'down'.",
    )
    assert num > 0, Logger.print(
        Logger.ERROR,
        msg=f"Invalid edge number: {num}. Must be greater than 0.",
    )
    collection.iv_params.edge_num = num
    collection.iv_params.edge_dir = dir


def loadCurveSets(collection: DataCollection):
    Logger.print(
        Logger.INFO,
        Logger.START,
        f"IVEval.loadCurveSet()",
    )

    params = collection.params
    header = params.selected_measurement

    Logger.print(Logger.DEBUG, msg=f"Loading IV data")
    # region load raw data & small calcs
    set = DataSet(name="adwin")

    file, mgroup = Files.open_file_group(
        collection.data,
        "/measurement/" + header.to_string() + "/" + params.selected_dataset + "/",
    )
    mgroup = Files.ensure_group(mgroup)
    offset = np.array(mgroup.get("offset/adwin"))

    set.voltage_offsets = (
        float(np.nanmean(np.array(offset["V1"]))),
        float(np.nanmean(np.array(offset["V2"]))),
    )

    sweep = np.array(mgroup.get("sweep/adwin"))
    trigger = np.array(sweep["trigger"], dtype="int")
    time = np.array(sweep["time"], dtype="float64")
    time = time - time[0]  # shift start time to 0
    v1 = np.array(sweep["V1"], dtype="float64")
    v2 = np.array(sweep["V2"], dtype="float64")

    v = (v1 - set.voltage_offsets[0]) / params.volt_amp[0]
    i = (v2 - set.voltage_offsets[1]) / params.volt_amp[1] / params.ref_resistor

    set.curves["voltage"] = v
    set.curves["current"] = i
    set.curves["time"] = time
    set.curves["trigger"] = trigger

    if file:
        file.close()
    # endregion
    collection.evaluation.raw_sets["adwin"] = set

    Logger.print(Logger.DEBUG, msg=f"Loading temperature data")
    # region get temperature data
    set = DataSet(name="bluefors")
    if collection.iv_config.eva_temperature:
        file, group = Files.open_file_group(
            collection.data,
            f"/measurement/{header.to_string()}/{params.selected_dataset}/sweep",
        )
        file = Files.ensure_file(file)
        group = Files.ensure_group(group)
        dset = Files.ensure_dataset(group.get("bluefors"))

        set.curves["temperature"] = np.array(dset["Tsample"], dtype="float64")
        time = np.array(dset["time"], dtype="float64")
        set.curves["time"] = time - time[0]  # shift start time to 0

        if file:
            file.close()

        # endregion
        collection.evaluation.raw_sets["bluefors"] = set

    set.check_curves()
    collection.evaluation.cached_sets = collection.evaluation.raw_sets.copy()


def filter_curve_sets(collection: DataCollection, group: str, bin_count: int = 0):
    Logger.print(
        Logger.INFO,
        Logger.START,
        f"IVEval.filter_curve_sets(bin_count={bin_count})",
    )
    source = collection.evaluation.cached_sets

    cache = source[group].copy()
    temp = cache.filter_trigger(collection.iv_params)
    if temp:
        cache = temp
    else:
        Logger.print(
            Logger.WARNING,
            msg=f"Skipped trigger selection for {group}. No trigger data found in the dataset.",
        )

    if bin_count != 0:
        start_time = cache.curves["time"][0]
        stop_time = cache.curves["time"][-1]

        t_bins = np.linspace(start_time, stop_time, bin_count)
        cache.curves["time-bin"] = t_bins

        if group == "adwin":
            v_bins = np.linspace(
                np.min(cache.curves["voltage"]),
                np.max(cache.curves["voltage"]),
                bin_count,
            )
            i_bins = np.linspace(
                np.min(cache.curves["current"]),
                np.max(cache.curves["current"]),
                bin_count,
            )

            binned_it, _ = Binning.bin(
                cache.curves["time"],
                cache.curves["current"],
                t_bins,
            )
            binned_vt, _ = Binning.bin(
                cache.curves["time"],
                cache.curves["voltage"],
                t_bins,
            )
            binned_iv, _ = Binning.bin(
                cache.curves["voltage"],
                cache.curves["current"],
                v_bins,
            )
            binned_vi, _ = Binning.bin(
                cache.curves["current"],
                cache.curves["voltage"],
                i_bins,
            )

            cache.curves["current"] = binned_it
            cache.curves["voltage"] = binned_vt
            cache.curves["current-voltage"] = binned_iv
            cache.curves["voltage-current"] = binned_vi
            cache.curves["voltage-bin"] = v_bins
            cache.curves["current-bin"] = i_bins
        elif group == "bluefors":
            binned_temp, _ = Binning.bin(
                cache.curves["time"],
                cache.curves["temperature"],
                t_bins,
            )
            cache.curves["temperature"] = binned_temp

    else:
        Logger.print(
            Logger.INFO,
            msg="Binning is disabled. No filter frequency set.",
        )

    source[group] = cache

    collection.evaluation.filtered_sets[group] = cache.copy()


def process_curve_sets(
    collection: DataCollection,
):
    Logger.print(
        Logger.INFO,
        Logger.START,
        f"IVEval.eval_loaded_curve_set()",
    )
    set = collection.evaluation.cached_sets

    conductance_quantum = constants.physical_constants["conductance quantum"][0]

    diff_conductance = (
        np.gradient(set["adwin"].curves["current"], set["adwin"].curves["voltage"])
        / conductance_quantum
    )
    diff_resistance = np.gradient(
        set["adwin"].curves["voltage"], set["adwin"].curves["current"]
    )
    temp = np.nanmean(set["bluefors"].curves["temperature"])

    diffs = DataSet()
    diffs.curves["diff_conductance"] = diff_conductance
    diffs.curves["diff_resistance"] = diff_resistance
    diffs.curves["temperature"] = temp

    collection.evaluation.cached_sets["diffs"] = diffs


def get_noise(
    collection: DataCollection, set_name: str, curve: tuple[str, str], result: str
):
    Logger.print(
        Logger.INFO,
        Logger.START,
        f"IVEval.get_noise({set_name}, {curve}, {result})",
    )
    source = collection.evaluation.cached_sets
    dataset = source[set_name]

    c1 = dataset.curves[curve[0]]
    c2 = dataset.curves[curve[1]]

    n = len(c1)
    dx = float(np.median(np.diff(c1)))
    freqs = np.fft.fftfreq(n, dx)[: n // 2]
    spectrum = np.abs(np.fft.fft(c2)[: n // 2])
    spec_log = np.log10(spectrum)

    if not "noise" in source.keys():
        source["noise"] = DataSet()

    source["noise"].curves[f"{result}-freq"] = freqs
    source["noise"].curves[f"{result}-spec"] = spec_log

    collection.evaluation.processed_sets["noise"] = source["noise"].copy()
