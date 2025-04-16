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
from typing import Dict, Any, List

# 3rd party
import numpy as np
import matplotlib.pyplot as plt
from h5py import File, Group, Dataset

# local
from integration.files import DataCollection
import utilities.logging as Logger

# endregion


@dataclass
class MeasurementHeader:
    name: str = field(default_factory=str)  # the measurement name
    variable: str = field(default_factory=str)  # the variied parameter
    constants: List[tuple] = field(
        default_factory=lambda: []
    )  # the constant parameters tuple (parameter, value, unit)
    varied_range: tuple = field(
        default_factory=lambda: (0.0, 0.0, 0.0)
    )  # min, max, step

    def to_string(self) -> str:
        return f"{self.name}: ({self.variable}) {self.varied_range} [{self.constants}]"

    @staticmethod
    def from_string(header: str) -> "MeasurementHeader":

        parts = header.split(" ")

        if parts[0].startswith("vna"):
            name = ""
            variable = parts[0]
        else:
            name = parts[0]
            variable = parts[1]

        def map_const(
            part: str,
        ) -> (
            tuple
        ):  # map sth from "constDecl_+0.000Unit" to ("constDecl", +0.000, "Unit")

            subparts = part.split("_")
            decl = subparts[0]

            val_arg = subparts[1]

            value, unit = MeasurementHeader.parse_number(val_arg)

            return (decl, value, unit)

        constants = []

        for part in parts[1:]:
            if part.startswith("vna"):
                part = part.split("_")
                constants.append(map_const("vna_" + part[1]))
                constants.append(map_const("vna_" + part[2]))
            else:
                constants.append(map_const(part))

        return MeasurementHeader(name, variable, constants, ())

    @staticmethod
    def parse_number(s: str) -> tuple:  # "Sign0.01Unit" -> (0.01, "Unit")
        sign = -1 if s[0] == "-" else 1

        def isDigit(s: str) -> bool:
            return s.isdigit() or s == "."

        s_str = "".join(filter(isDigit, s[1:]))
        value = float(s_str) * sign if not s_str == "" else np.nan
        if s_str == "":
            offset = 3
        if not s_str == "":
            offset = 1 + len(s_str)

        unit = s[offset:]
        return value, unit


@dataclass
class Parameters:
    volt_amp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    ref_resistor: float = field(default_factory=lambda: 51.689e3)
    selected_measurement: MeasurementHeader = field(
        default_factory=lambda: MeasurementHeader()
    )
    available_measurements: List[MeasurementHeader] = field(default_factory=lambda: [])

    def __setattr__(self, name: str, value: Any):
        if name == "volt_amp" and not np.array_equal(value, np.array([0.0, 0.0])):
            Logger.print(Logger.DEBUG, msg=f"Params.{name} = {value}")
        object.__setattr__(self, name, value)


@dataclass
class Curve:
    headers: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    norm: float = field(default_factory=float)


def setup(collection: DataCollection):

    collection.packets["params"] = Parameters()

    name = collection.data.name
    Logger.print(Logger.INFO, Logger.START, "GenEval.setup()")


# class Measurements(Enum):


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
        keys = list(data_file.keys())
        Logger.print(Logger.INFO, msg=f"Available measurements: {keys}")


def showLoadedMeasurements(collection: DataCollection):
    """
    Displays the available measurement types, currently loaded.
    """
    Logger.print(Logger.INFO, Logger.START, "GenEval.showLoadedMeasurements()")

    num_headers = len(collection.params.available_measurements)
    num_digits = len(str(num_headers))
    for i, header in enumerate(collection.params.available_measurements):
        Logger.print(
            Logger.INFO,
            msg=f"[{i+1:>{num_digits}}] {header.to_string()}",
        )


def loadMeasurements(
    collection: DataCollection,
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
                subgroup = measurement_group[measurement_name]

                if isinstance(subgroup, Group):
                    header = MeasurementHeader.from_string(measurement_name)

                    params = []
                    for subkey in subgroup.keys():
                        params.append(MeasurementHeader.parse_number(subkey))

                    min = np.min(list(map(lambda x: x[0], params)))
                    max = np.max(list(map(lambda x: x[0], params)))
                    step = params[1][0] - params[0][0]
                    header.varied_range = (min, max, step)

                    headers.append(header)

        collection.params.available_measurements = headers


def select_measurement(collection: DataCollection, header_index: int):
    Logger.print(
        Logger.INFO,
        Logger.START,
        msg=f"GenEval.select_measurement(index={header_index})",
    )

    collection.params.selected_measurement = collection.params.available_measurements[
        header_index - 1
    ]

    Logger.print(
        Logger.DEBUG,
        msg=f"Selected: {collection.params.selected_measurement.to_string()}",
    )
