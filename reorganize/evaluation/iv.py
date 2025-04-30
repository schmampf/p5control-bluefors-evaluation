# region imports
# std
from dataclasses import dataclass, field
from typing import Any

# third-party
import numpy as np
import h5py

# local
import integration.files as Files
from integration.files import DataCollection
import utilities.logging as Logger

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
    curves: dict[str, Any] = field(default_factory=dict)
    voltage_offsets: tuple[float, float] = field(default_factory=tuple)


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
    temperatures: Axis = field(default_factory=Axis)  # one per dataset
    time_starts: Axis = field(default_factory=Axis)  # one per dataset
    time_stops: Axis = field(default_factory=Axis)  # one per dataset
    temp_current: Map = field(default_factory=Map)
    temp_voltage: Map = field(default_factory=Map)
    diff_resistance: Map = field(default_factory=Map)
    diff_conductance: Map = field(default_factory=Map)


@dataclass
class Results:
    upsweep: Result = field(default_factory=Result)
    downsweep: Result = field(default_factory=Result)
    cache: Result = field(default_factory=Result)


@dataclass
class Configuration:
    eva_current: bool = True
    eva_voltage: bool = True
    eva_temperature: bool = True
    eva_even_spaced: bool = False


@dataclass
class Parameters:
    downsample_freq: int = 3
    upsample_voltage: int = 137
    upsample_current: int = 137
    upsample_temperature: int = 0
    upsample_amplitude: int = 0
    edge_num: int = 0
    edge_dir: str = field(default="nan")


# endregion


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


def setup(bib: DataCollection):
    bib.iv_config = Configuration()
    bib.iv_params = Parameters()
    bib.evaluation = Results()

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


def loadCurveSet(collection: DataCollection):
    Logger.print(
        Logger.INFO,
        Logger.START,
        f"IVEval.loadCurveSet()",
    )

    set = DataSet()
    params = collection.params
    header = params.selected_measurement

    # region raw iv loading
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

    if file:
        file.close()
    # endregion

    if collection.iv_config.eva_temperature:
        file, group = Files.open_file_group(
            collection.data,
            f"/measurement/{header.to_string()}/{params.selected_dataset}/sweep/",
        )
        file = Files.ensure_file(file)
        group = Files.ensure_group(group)
        check = "bluefors" in group.keys()
        if check:
            temperature = group.get("bluefors/Tsample")
            time = group.get("bluefors/time")
        else:
            temperature = np.array(file.get("status/bluefors/temperature/8-mcbj"))["T"]
            time = np.array(file.get("status/bluefors/temperature/8-mcbj"))["time"]
        set.curves["temperature"] = np.array(temperature, dtype="float64")
        set.curves["time"] = np.array(time, dtype="float64")
