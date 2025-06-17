"""
Evaluation Module for p5control-bluefors Data Processing

This module provides an evaluation class (`Evaluation`) to analyze measurement
data recorded using the p5control-bluefors system. It is designed to work with HDF5
files and facilitates accessing, processing, and visualizing measurement data.

Main Features:
---------------
- **File Handling**:
  - Opens and reads HDF5 files.
  - Extracts available measurement data.
  - Lists available keys for specific measurements.

- **Measurement Processing**:
  - Allows selecting a specific measurement for analysis.
  - Extracts and processes relevant measurement keys.
  - Stores and updates measurement-related parameters.

- **Amplification Analysis**:
  - Retrieves voltage amplification values over time.
  - Plots amplification changes during measurements.
  - Sets user-defined amplification factors for further evaluation.

- **Logging and Debugging**:
  - Logs key steps and potential errors during file access and data processing.
  - Provides detailed debug information for troubleshooting.

Usage:
------
This class serves as a base for more specific evaluation routines.
A typical workflow might include:

1. **Initialize the evaluation instance**:
   ```python
   evaluator = BaseEvaluation(name="Test Evaluation")
"""

# region imports
# std lib
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

# 3rd party
import numpy as np
import matplotlib.pyplot as plt
from h5py import File, Group, Dataset

# local
import integration.files as Files
from integration.files import DataCollection
import utilities.logging as Logger
from utilities.name_gen import format_scientific as fmt

# endregion


# region Data Classes
@dataclass
class Range:
    min: float = field(default_factory=float)
    max: float = field(default_factory=float)
    step: float = field(default_factory=float)


@dataclass
class Variable:
    name: str = field(default_factory=str)
    range: Range = field(default_factory=Range)
    unit: str = field(default_factory=str)


@dataclass
class Constant:
    name: str = field(default_factory=str)
    value: float = field(default_factory=float)
    unit: str = field(default_factory=str)


@dataclass
class MeasurementHeader:
    name: str = field(default_factory=str)
    variable: Variable = field(default_factory=Variable)
    constants: List[Constant] = field(default_factory=list)

    @staticmethod
    def unit_to_symbol(unit: str) -> str:
        match unit:
            case "V":
                return "U"
            case "A":
                return "I"
            case "Hz":
                return "f"
            case "T":
                return "H"
            case "K":
                return "T"
            case "W":
                return "P"
            case _:
                return "None"

    def to_string(self) -> str:

        title = self.name + ": " if not self.name == "" else ""
        var_str = f"({self.variable.name},({fmt(self.variable.range.min)},{fmt(self.variable.range.max)}{","+fmt(self.variable.range.step) if not self.variable.range.step == 0.0 else ""}),{self.variable.unit})"
        const_str = ",".join(
            f"({c.name},{fmt(c.value)},{c.unit})" for c in self.constants
        )
        return f"{title}var={var_str} const=[{const_str}]"

    @staticmethod
    def from_string(header: str) -> "MeasurementHeader":
        result = MeasurementHeader()

        parts = header.split(" ")
        nextindex = 1
        if ":" in parts[0]:
            title = parts[0][0:-1]
            result.name = title
        else:
            # if no title is given, skip title parsing
            nextindex = 0
        var = parts[nextindex][4:-1]
        const = parts[nextindex + 1][7:-1]

        var = var.replace("(", "").replace(")", "").split(",")
        result.variable = Variable(
            name=var[0],
            range=Range(
                min=float(var[1]),
                max=float(var[2]),
                step=float(var[3]) if len(var) == 5 else 0.0,
            ),
            unit=var[4] if len(var) == 5 else var[3],
        )

        const = const.split("),(")
        for i, part in enumerate(const):
            const[i] = part.replace("(", "").replace(")", "")

        if not const == [""]:
            res_const = []
            for part in const:
                subparts = part.split(",")
                res_const.append(
                    Constant(
                        name=subparts[0],
                        value=float(subparts[1]),
                        unit=subparts[2],
                    )
                )
            result.constants = res_const

        return result

    @staticmethod
    def parse_number(s: str) -> tuple:  # "Sign0.01Unit" -> (0.01, "Unit")
        if "=" in s:
            s = s.split("=")[1]
        s = s.replace(" ", "")

        def isDigit(s: str) -> bool:
            return s.isdigit() or s == "."

        def isNumber(s: str) -> bool:
            return isDigit(s) or s in ["+", "-"]

        num_start = 0
        num_end = 0
        # walk from left to right until a number char is found
        for i, c in enumerate(s):
            if isNumber(c):
                num_start = i
                break
        # walk from right to left until a number char is found
        for i, c in enumerate(s[::-1]):
            if isNumber(c):
                num_end = i
                break

        num_str = s[num_start : len(s) - num_end]

        signed = True if num_str[0] in ["+", "-"] else False
        sign = -1 if num_str[0] == "-" else 1
        num_str = num_str[1:] if signed else num_str

        if num_str == "NAN":
            num_str = "0.0"

        value = float(num_str) * sign if not num_str == "" else np.nan

        unit = s[len(s) - num_end :]

        return value, unit


@dataclass
class Parameters:
    volt_amp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    ref_resistor: float = field(default_factory=lambda: 51.689e3)
    selected_measurement: MeasurementHeader = field(
        default_factory=lambda: MeasurementHeader()
    )
    selected_dataset: str = field(default_factory=lambda: "")

    available_measurements: List[MeasurementHeader] = field(default_factory=lambda: [])
    available_measurement_entries: int = field(default_factory=lambda: -1)
    available_measurement_entries_labels: List[str] = field(
        default_factory=lambda: [],
    )

    linearizeYAxis: bool = field(default_factory=lambda: False)
    evalTemperature: bool = field(default_factory=lambda: False)
    normalizeXAxis: bool = field(default_factory=lambda: True)
    smoothData: bool = field(default_factory=lambda: True)

    def __setattr__(self, name: str, value: Any):
        if name == "volt_amp" and not np.array_equal(value, np.array([0.0, 0.0])):
            Logger.print(Logger.DEBUG, msg=f"Params.{name} = {value}")
        object.__setattr__(self, name, value)


# endregion Data Classes


def setup(collection: DataCollection):

    collection.packets["params"] = Parameters()

    name = collection.data.name
    Logger.print(Logger.INFO, Logger.START, "GenEval.setup()")


def showAmplification(collection: DataCollection):
    """
    Plots the voltage amplification over time for the entire measurement.
    """
    data = collection.packets["data"]
    Logger.print(Logger.INFO, Logger.START, "GenEval.showAmplifications()")
    file_name = f"{data.file_directory}{data.file_folder}{data.file_name}"

    # check if file exists
    if not os.path.exists(file_name):
        Logger.print(Logger.ERROR, msg=f"Error: File does not exist: {file_name}")
        return

    with File(file_name, "r") as data_file:

        if data_file.__contains__("status/femto"):
            femto_key = "femto"
        elif data_file.__contains__("status/femtos"):
            femto_key = "femtos"
        else:
            Logger.print(Logger.ERROR, msg="Femto status not found.")
            return

        femto_data = np.array(data_file[f"status/{femto_key}"])

    plt.close(1000)
    plt.figure(1000, figsize=(6, 1.5))
    plt.semilogy(
        femto_data["time"],
        femto_data["amp_A"],
        "-",
        label="Voltage Amplification 1",
    )
    plt.semilogy(
        femto_data["time"],
        femto_data["amp_B"],
        "--",
        label="Voltage Amplification 2",
    )
    plt.legend()
    plt.title("Femto Amplifications (Status)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplification")
    plt.tight_layout()
    plt.show()


def showFileMeasurements(collection: DataCollection):
    """
    Displays the available measurement keys in the HDF5 file.
    """
    data = collection.packets["data"]
    Logger.print(Logger.INFO, Logger.START, "GenEval.showFileMeasurements()")
    file_name = f"{data.file_directory}{data.file_folder}{data.file_name}"

    # check if file exists
    if not os.path.exists(file_name):
        # logger.error("(%s) Error: File does not exist: %s", data.name, file_name)
        Logger.print(Logger.ERROR, msg=f"Error: File does not exist: {file_name}")
        return

    # show selected file
    Logger.print(Logger.DEBUG, msg=f"Opening file: {file_name}")

    with File(file_name, "r") as data_file:
        group = data_file.get("measurement")
        if group and isinstance(group, Group):
            keys = list(group.keys())
            for key in keys:
                Logger.print(Logger.INFO, msg=f"Available measurements: {key}")


def showLoadedMeasurements(collection: DataCollection):
    """
    Displays the available measurement types, currently loaded.
    """
    Logger.print(Logger.INFO, Logger.START, "GenEval.showLoadedMeasurements()")

    num_headers = len(collection.params.available_measurements)

    if num_headers == 0:
        Logger.print(Logger.ERROR, msg="No measurements loaded.")
        return

    num_digits = len(str(num_headers))
    for i, header in enumerate(collection.params.available_measurements):
        Logger.print(
            Logger.INFO,
            msg=f"[{i+1:>{num_digits}}] {header.to_string()}",
        )


def loadMeasurements(
    collection: DataCollection,
    findRange: bool = False,
):
    """
    Loads the specified measurement type and key from the HDF5 file.
    """
    data = collection.data
    Logger.print(Logger.INFO, Logger.START, "GenEval.loadMeasurements()")

    # check if file exists
    file_name = f"{data.file_directory}{data.file_folder}{data.file_name}"
    if not os.path.exists(file_name):
        Logger.print(Logger.ERROR, msg=f"Error: File does not exist: {file_name}")
        return

    # show selected file
    Logger.print(Logger.DEBUG, msg=f"Opening file: {file_name}")

    with File(file_name, "r") as data_file:
        # get available measurements from file
        measurement_group = data_file.get("measurement")

        headers = []

        if measurement_group and isinstance(measurement_group, Group):
            for measurement_name in measurement_group.keys():
                try:
                    header = MeasurementHeader.from_string(measurement_name)

                    if findRange:
                        Logger.print(
                            Logger.INFO,
                            msg=f"Attempting to parse measurement range for: {measurement_name}",
                        )
                        Logger.print(
                            Logger.INFO,
                            msg=f"  The Result may be inaccurate.",
                        )
                        variations = measurement_group.get(measurement_name)
                        if isinstance(variations, Group):
                            labels = variations.keys()
                            nums = []
                            for label in labels:
                                try:
                                    num = MeasurementHeader.parse_number(label)
                                    nums.append(num)
                                except Exception as e:
                                    Logger.print(
                                        Logger.DEBUG,
                                        msg=f"Measurement variation format unknown: {label} - {e}",
                                    )
                                    continue

                            nums = sorted(nums, key=lambda x: x[0])

                            min = nums[0][0]
                            max = nums[-1][0]
                            step = nums[1][0] - nums[0][0]

                            declared_decimals = len(str(nums[0][0]).split(".")[1])
                            step = round(step, declared_decimals)

                            header.variable.range = Range(
                                min=min,
                                max=max,
                                step=step,
                            )
                    headers.append(header)
                except Exception as e:
                    Logger.print(
                        Logger.ERROR,
                        msg=f"Measurement header format unknown: {measurement_name} - {e}",
                    )
                    continue

        collection.params.available_measurements = headers


def select_measurement(collection: DataCollection, header_index: int):
    Logger.print(
        Logger.INFO,
        Logger.START,
        msg=f"GenEval.select_measurement(index={header_index})",
    )

    num_available_measurements = len(collection.params.available_measurements)

    if num_available_measurements == 0:
        Logger.print(Logger.ERROR, msg="No measurements loaded.")
        return

    if num_available_measurements < header_index:
        Logger.print(
            Logger.ERROR,
            msg=f"Error: Index out of range. Available measurements: {num_available_measurements}",
        )
        return

    header = collection.params.available_measurements[header_index - 1]
    collection.params.selected_measurement = header
    file, group = Files.open_file_group(
        collection.data, "/measurement/" + header.to_string()
    )
    if isinstance(group, Group):
        collection.params.available_measurement_entries = len(group.keys())
        collection.params.available_measurement_entries_labels = [
            str(s) for s in group.keys()
        ]
    if file:
        file.close()

    Logger.print(
        Logger.INFO,
        msg=f"Selected: {collection.params.selected_measurement.to_string()}",
    )
    Logger.print(
        Logger.INFO,
        msg=f"    with: {collection.params.available_measurement_entries} entries",
    )


def select_CurveSet(collection: DataCollection, var_range_index: int):
    """
    Loads the specified measurement set (Sweep and Offset) from the HDF5 file.
    If the NAN measurement is present, it will be loaded first.
    """
    data = collection.data
    Logger.print(
        Logger.INFO,
        Logger.START,
        f"GenEval.select_CurveSet(var_range_index={var_range_index})",
    )

    header = collection.params.selected_measurement
    file, group = Files.open_file_group(data, "/measurement/" + header.to_string())

    if group and isinstance(group, Group):
        subgroups = group.keys()

        indexes = []
        nums = []
        nan_index = -1

        for i, label in enumerate(subgroups):
            if "NAN" in label:
                nan_index = i
                continue

            try:
                num = MeasurementHeader.parse_number(label)
                nums.append(num[0])
                indexes.append(i)
            except Exception as e:
                Logger.print(
                    Logger.DEBUG,
                    msg=f"Measurement variation format unknown: {label} - {e}",
                )
                continue

        # numerical sort both indexes and nums depending on nums
        # nums = sorted(nums, key=lambda x: x[0])
        nums, indexes = zip(*sorted(zip(nums, indexes)))
        nums = list(nums)
        indexes = list(indexes)

        indexes = (
            [nan_index] + indexes if nan_index != -1 else indexes
        )  # append nan_index to the front of the list

        selected_index = indexes[var_range_index]
        keys = list(subgroups)
        selected_key = keys[selected_index]

        Logger.print(
            Logger.INFO,
            msg=f"Selected DataSet: {selected_key}",
        )
        collection.params.selected_dataset = selected_key

    if file:
        file.close()


def get_selection_path(collection: DataCollection) -> str:
    """
    Returns the path to the selected measurement in the HDF5 file.
    """
    header = collection.params.selected_measurement
    return f"/measurement/{header.to_string()}/{collection.params.selected_dataset}"
