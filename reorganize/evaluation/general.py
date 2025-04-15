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
from h5py import File

# local
from integration.files import DataCollection
from utilities.logger import logger

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
        return f"{self.name}: ({self.variable}) [{self.constants}] {self.varied_range}"

    @staticmethod
    def fromString(string: str) -> "MeasurementHeader":

        parts = string.split(":")
        name = parts[0].strip()

        variable, constants, varied_range = parts[1].split("[")
        variable = variable.strip()

        constants = constants.strip("]").split(",")
        constants = [tuple(map(str.strip, c.split("="))) for c in constants]

        varied_range = varied_range.strip(")").split(",")
        varied_range = tuple(map(float, varied_range))

        return MeasurementHeader(name, variable, constants, varied_range)


@dataclass
class Parameters:
    volt_amp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))

    ref_resistor: float = field(default_factory=lambda: 51.689e3)
    selected_measurement: MeasurementHeader = field(
        default_factory=lambda: MeasurementHeader()
    )
    available_measurements: List[MeasurementHeader] = field(default_factory=lambda: [])


@dataclass
class Curve:
    headers: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    norm: float = field(default_factory=float)


def setup(collection: DataCollection):

    collection.packets["params"] = Parameters()

    name = collection.data.name
    logger.info("(%s) + Setup GenEval Completed!", name)


# class Measurements(Enum):


# def select_measurement(collection: DataCollection, header: MeasurementHeader):
#     """
#     Select a specific measurement for evaluation.
#     """
#     logger.debug("(%s) select_measurement()", collection.packets["data"].name)


def showAmplification(collection: DataCollection):
    """
    Plots the voltage amplification over time for the entire measurement.
    """
    data = collection.packets["data"]
    logger.debug("(%s) showAmplifications()", data.name)
    file_name = f"{data.file_directory}{data.file_folder}{data.file_name}"

    # check if file exists
    if not os.path.exists(file_name):
        logger.error("(%s) Error: File does not exist: %s", data.name, file_name)
        return

    with File(file_name, "r") as data_file:

        if data_file.__contains__("status/femto"):
            femto_key = "femto"
        elif data_file.__contains__("status/femtos"):
            femto_key = "femtos"
        else:
            logger.error("(%s) ...femto(s) status not found.", data.name)
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
    logger.debug("(%s) showMeasurements()", data.name)
    file_name = f"{data.file_directory}{data.file_folder}{data.file_name}"

    # check if file exists
    if not os.path.exists(file_name):
        logger.error("(%s) Error: File does not exist: %s", data.name, file_name)
        return

    # show selected file
    logger.info("(%s) Opening file: %s", data.name, file_name)

    with File(file_name, "r") as data_file:
        keys = list(data_file.keys())
        logger.info("Available measurements: %s", keys)


def showLoadedMeasurements(collection: DataCollection):
    """
    Displays the available measurement types, currently loaded.
    """
    data = collection.packets["data"]
    logger.debug("(%s) showMeasurements()", data.name)
    logger.info(
        "(%s) Available measurements: %s", data.name, data.available_measurements
    )


def loadMeasurements(
    collection: DataCollection,
):
    """
    Loads the specified measurement type and key from the HDF5 file.
    """
    data = collection.packets["data"]
    logger.debug("(%s) loadMeasurements()", data.name)

    # check if file exists
    file_name = f"{data.file_directory}{data.file_folder}{data.file_name}"
    if not os.path.exists(file_name):
        logger.error("(%s) Error: File does not exist: %s", data.name, file_name)
        return

    # show selected file
    logger.info("(%s) Opening file: %s", data.name, file_name)

    with File(file_name, "r") as data_file:
        # get available measurements from file
        measurement_group = data_file.get("measurement")
        if isinstance(measurement_group, File):
            measurements = measurement_group.keys()
        else:
            logger.error(
                "(%s) 'measurement' is not a group or does not exist.", data.name
            )
            return

        # logger.info("Available measurements: %s", keys)
