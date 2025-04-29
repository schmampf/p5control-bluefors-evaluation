# region imports
# std
from dataclasses import dataclass, field

# third-party
import numpy as np

# local
import integration.files as Files
from integration.files import DataCollection
import utilities.logging as Logger

# endregion


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
    curves: dict[str, Curve] = field(default_factory=dict)
    voltage_offsets: list[tuple[float, float]] = field(default_factory=list)


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
    temperatures: Axis = field(default_factory=Axis)
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
    downsample_freq = 3
    upsample_voltage = 137
    upsample_current = 137
    upsample_temperature = 0
    upsample_amplitude = 0


def setup(bib: DataCollection):
    bib.iv_config = Configuration()
    bib.iv_params = Parameters()
    bib.evaluation = Results()

    Logger.print(Logger.INFO, Logger.START, "IVEval.setup()")
