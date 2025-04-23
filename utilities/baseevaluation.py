"""
Base Evaluation Module for p5control-bluefors Data Processing

This module provides a base evaluation class (`BaseEvaluation`) to analyze measurement
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

import logging
import numpy as np
import matplotlib.pyplot as plt
from h5py import File

from utilities.baseclass import BaseClass
from utilities.key_database import POSSIBLE_MEASUREMENT_KEYS

logger = logging.getLogger(__name__)


class BaseEvaluation(BaseClass):
    """
    A base class for evaluating measurement data from p5control-bluefors.

    This class provides methods to:
    - Access and read HDF5 files.
    - Display and set measurement keys.
    - Show and set amplification values.
    - Extract specific data from measurements.
    """

    def __init__(
        self,
        name: str = "base eva",
    ):
        """
        Initializes the BaseEvaluation class.

        Parameters:
        - name (str): A name for the evaluation instance.
        """
        super().__init__()
        self._base_eva_name = name

        # initialize key lists
        self.possible_measurement_keys = POSSIBLE_MEASUREMENT_KEYS

        # initialize base_evaluation stuff
        self.base_evaluation = {
            "voltage_amplification_1": 1,
            "voltage_amplification_2": 1,
            "reference_resistor": 51.689e3,
            "index_trigger_up": 1,
            "index_trigger_down": 2,
            "measurement_key": "",
            "specific_keys": [],
            "y_0_key": "",
            "y_unsorted": np.array([]),
        }

        logger.info("(%s) ... BaseEvaluation initialized.", self._base_eva_name)

    def showAmplifications(self) -> None:
        """
        Plots the voltage amplification over time for the entire measurement.
        """
        logger.debug("(%s) showAmplifications()", self._base_eva_name)
        file_name = f"{self.file_directory}{self.file_folder}{self.file_name}"

        with File(file_name, "r") as data_file:

            if data_file.__contains__("status/femto"):
                femto_key = "femto"
            elif data_file.__contains__("status/femtos"):
                femto_key = "femtos"
            else:
                logger.error("(%s) ...femto(s) status not found.", self._base_eva_name)
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

    def setAmplifications(
        self, voltage_amplification_1: int, voltage_amplification_2: int
    ) -> None:
        """
        Sets the amplification factors for voltage calculations.

        Parameters
        ----------
        voltage_amplification_1: int
            Amplification factor for channel 1.
        voltage_amplification_2: int
            Amplification factor for channel 2.
        """
        logger.debug(
            "(%s) setAmplifications(%d, %d)",
            self._base_eva_name,
            voltage_amplification_1,
            voltage_amplification_2,
        )
        self.voltage_amplification_1 = voltage_amplification_1
        self.voltage_amplification_2 = voltage_amplification_2

    def showMeasurements(self) -> None:
        """
        Lists all available measurements in the file.
        """
        logger.debug("(%s) showMeasurements()", self._base_eva_name)

        file_name = self.file_directory + self.file_folder + self.file_name
        with File(file_name, "r") as data_file:
            measurements = list(data_file["measurement"].keys())  # type: ignore
            logger.info(
                "(%s) Available measurements: %s", self._base_eva_name, measurements
            )

    def setMeasurement(self, measurement_key: str) -> None:
        """
        Sets the measurement key, which is required for further evaluation.

        Parameters
        ----------
        measurement_key: str
            The name of the measurement.

        Raises:
        - KeyError: If the given key does not exist in the file.
        """
        logger.debug("(%s) setMeasurement('%s')", self._base_eva_name, measurement_key)

        file_name = self.file_directory + self.file_folder + self.file_name
        with File(file_name, "r") as data_file:
            try:
                measurement_data = data_file.get(f"measurement/{measurement_key}")
                self.specific_keys = list(measurement_data)  # type: ignore
                self.measurement_key = measurement_key
            except KeyError:
                self.specific_keys = []
                self.measurement_key = ""
                logger.error(
                    "(%s) Measurement key '%s' not found in file.",
                    self._base_eva_name,
                    measurement_key,
                )

    def showKeys(self) -> None:
        """
        Displays a subset of available measurement keys.
        """
        logger.debug("(%s) showKeys()", self._base_eva_name)
        preview_keys = self.specific_keys[:2] + self.specific_keys[-2:]
        logger.info(
            "(%s) Measurement keys preview: %s", self._base_eva_name, preview_keys
        )

    def addKey(self, key: str, y_value: float) -> None:
        """
        Adds a key to the specific key list and updates the y-values.

        Parameters
        ----------
        key : str
            The measurement key to add.
        y_value : float
            The corresponding y-value to store.
        """
        logger.info("(%s) addKey('%s', %f)", self._base_eva_name, key, y_value)

        self.specific_keys.append(key)
        self.y_unsorted = np.concatenate((self.y_unsorted, [y_value]))

    def setKeys(
        self,
        index_0: int = 0,
        index_1=None,
        norm: float = 0,
        to_pop: str = "",
    ):
        """
        Sets Keys and calculate y_unsorted. Mandatory for further Evaluation.

        Parameters
        ----------
        parameters : str
            [
            index of first y-value,
            index of last y-value,
            normalization for y-value,
            key to pop,
            ]
        index_0 : int
            index of first y_value
        index_1
            index of last y_value
        norm : float
            normalization for y-value
        to_pop : str
            key to pop
        """
        logger.info(
            "(%s) setKeys(%s, %s, %s, %s)",
            self._base_eva_name,
            index_0,
            index_1,
            norm,
            to_pop,
        )
        if self.measurement_key == "":
            logger.warning("(%s) Do setMeasurement() first.", self._base_eva_name)
            return

        if to_pop != "":
            try:
                self.specific_keys.remove(to_pop)
                self.y_0_key = to_pop
            except KeyError:
                logger.warning("(%s) Key to pop is not found.", self._base_eva_name)

        y = []
        for key in self.specific_keys:
            temp = key[index_0:index_1]
            temp = float(temp) * norm
            y.append(temp)
        y = np.array(y)

        self.y_unsorted = y

    ### base_evaluation Properties ###

    @property
    def voltage_amplification_1(self):
        """get voltage_amplification_1"""
        return self.base_evaluation["voltage_amplification_1"]

    @voltage_amplification_1.setter
    def voltage_amplification_1(self, voltage_amplification_1: int):
        """set voltage_amplification_1"""
        self.base_evaluation["voltage_amplification_1"] = voltage_amplification_1
        logger.debug(
            "(%s) voltage_amplification_1 = %s",
            self._base_eva_name,
            voltage_amplification_1,
        )

    @property
    def voltage_amplification_2(self):
        """get voltage_amplification_2"""
        return self.base_evaluation["voltage_amplification_2"]

    @voltage_amplification_2.setter
    def voltage_amplification_2(self, voltage_amplification_2: int):
        """set voltage_amplification_2"""
        self.base_evaluation["voltage_amplification_2"] = voltage_amplification_2
        logger.debug(
            "(%s) voltage_amplification_2 = %s",
            self._base_eva_name,
            voltage_amplification_2,
        )

    @property
    def reference_resistor(self):
        """get reference_resistor"""
        return self.base_evaluation["reference_resistor"]

    @reference_resistor.setter
    def reference_resistor(self, reference_resistor: float):
        """set reference_resistor"""
        self.base_evaluation["reference_resistor"] = reference_resistor
        logger.debug(
            "(%s) reference_resistor = %s", self._base_eva_name, reference_resistor
        )

    @property
    def index_trigger_up(self):
        """get index_trigger_up"""
        return self.base_evaluation["index_trigger_up"]

    @index_trigger_up.setter
    def index_trigger_up(self, index_trigger_up: int):
        """set index_trigger_up"""
        self.base_evaluation["index_trigger_up"] = index_trigger_up
        logger.debug(
            "(%s) index_trigger_up = %s", self._base_eva_name, index_trigger_up
        )

    @property
    def index_trigger_down(self):
        """get index_trigger_down"""
        return self.base_evaluation["index_trigger_down"]

    @index_trigger_down.setter
    def index_trigger_down(self, index_trigger_down: int):
        """set index_trigger_down"""
        self.base_evaluation["index_trigger_down"] = index_trigger_down
        logger.debug(
            "(%s) index_trigger_down = %s", self._base_eva_name, index_trigger_down
        )

    @property
    def measurement_key(self):
        """get measurement_key"""
        return self.base_evaluation["measurement_key"]

    @measurement_key.setter
    def measurement_key(self, measurement_key: str):
        """set measurement_key"""
        self.base_evaluation["measurement_key"] = measurement_key
        logger.debug("(%s) measurement_key = %s", self._base_eva_name, measurement_key)

    @property
    def specific_keys(self):
        """get specific_keys"""
        return self.base_evaluation["specific_keys"]

    @specific_keys.setter
    def specific_keys(self, specific_keys: list[str]):
        """set specific_keys"""
        self.base_evaluation["specific_keys"] = specific_keys
        logger.debug("(%s) specific_keys = %s", self._base_eva_name, specific_keys)

    @property
    def y_0_key(self):
        """get y_0_key"""
        return self.base_evaluation["y_0_key"]

    @y_0_key.setter
    def y_0_key(self, y_0_key: str):
        """set specific_keys"""
        self.base_evaluation["y_0_key"] = y_0_key
        logger.debug("(%s) y_0_key = %s", self._base_eva_name, y_0_key)

    @property
    def y_unsorted(self):
        """get y_unsorted"""
        return self.base_evaluation["y_unsorted"]

    @y_unsorted.setter
    def y_unsorted(self, y_unsorted: np.ndarray):
        """set y_unsorted"""
        self.base_evaluation["y_unsorted"] = y_unsorted
        logger.debug("(%s) y_unsorted = %s", self._base_eva_name, y_unsorted)


'''
    def accessFile(self) -> File:
        """
        Opens the HDF5 file for reading.

        Returns:
        - File: The opened HDF5 file object.

        Raises:
        - AttributeError: If the file cannot be found.
        """
        logger.debug("(%s) accessFile()", self._base_eva_name)
        try:
            file_name = self.file_directory + self.file_folder + self.file_name
            return File(file_name, "r")
        except AttributeError:
            logger.error("(%s) File cannot be found!", self._base_eva_name)
            raise


'''
