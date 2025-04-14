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

# 3rd party
import numpy as np
import matplotlib.pyplot as plt
import h5py

# local
from integration.files import DataCollection
from utilities.logger import logger
#endregion

@dataclass 
class MeasurementHeader:
    type: str = field(default_factory=str)

@dataclass
class Parameters:
    volt_amp: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))
    ref_resistor: float = field(default_factory=lambda: 51.689E3)
    selected_measurement: MeasurementHeader = field(default_factory=lambda: MeasurementHeader())


  
# class Measurements(Enum):
    
    

class EvaluationAPI:
    """
    EvaluationAPI provides methods to evaluate measurement data.
    It serves as a base class for specific evaluation routines.
    """
    
    @staticmethod
    def select_measurement(collection: DataCollection, header: MeasurementHeader):
        """
        Select a specific measurement for evaluation.
        """
        logger.debug("(%s) select_measurement()", collection.packets["data"].name)
    
    @staticmethod
    def showAmplification(collection: DataCollection):
        """
        Plots the voltage amplification over time for the entire measurement.
        """
        data = collection.packets["data"]
        logger.debug("(%s) showAmplifications()", data.name)
        file_name = f"{data.file_directory}{data.file_folder}{data.file_name}"

        with h5py.File(file_name, "r") as data_file:

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
        