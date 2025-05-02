# region imports
# std
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

# third-party
import numpy as np
import h5py

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
    raw_sets: dict[str, DataSet] = field(default_factory=dict)
    processed_sets: dict[str, DataSet] = field(default_factory=dict)
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


# class FilterModes(Enum):
#     TRIGGER = 1
#     FT = 2
#     BIN = 3


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
    collection.evaluation.raw_sets["adwin"] = set
    # endregion

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
        set.curves["time"] = np.array(dset["time"], dtype="float64")

        if file:
            file.close()

        collection.evaluation.raw_sets["bluefors"] = set
    # endregion

    set.check_curves()


def filter_curve_sets(collection: DataCollection):
    filtered = collection.evaluation.raw_sets["adwin"].filter_trigger(
        collection.iv_params
    )
    collection.evaluation.processed_sets["adwin"] = filtered

    start_time = filtered.curves["time"][0]
    stop_time = filtered.curves["time"][-1]

    bins = np.arange(start_time, stop_time, 43)

    binned_it, _ = Binning.bin(
        filtered.curves["time"],
        filtered.curves["current"],
        bins,
    )
    binned_vv, _ = Binning.bin(
        filtered.curves["time"],
        filtered.curves["voltage"],
        bins,
    )
    binned_iv, _ = Binning.bin(
        filtered.curves["voltage"],
        filtered.curves["current"],
        bins,
    )
    binned_vi, _ = Binning.bin(
        filtered.curves["current"],
        filtered.curves["voltage"],
        bins,
    )
    binned_t = bins

    filtered.curves["current-time"] = binned_it
    filtered.curves["voltage-time"] = binned_vv
    filtered.curves["current-voltage"] = binned_iv
    filtered.curves["voltage-current"] = binned_vi
    filtered.curves["time"] = binned_t


def process_curve_sets(
    collection: DataCollection,
):
    Logger.print(
        Logger.INFO,
        Logger.START,
        f"IVEval.eval_loaded_curve_set()",
    )
    config = collection.iv_config
    params = collection.iv_params
    set = collection.evaluation.upsweep
