"""
Module: IV_Evaluation

Description:
    This module provides a class for evaluating current-voltage (IV) characteristics.
    It includes methods for processing raw IV data, calculating differentials,
    and managing various IV-related properties such as voltage offsets, binning,
    and up/down sweeps. The class also includes property methods to access and
    modify evaluation parameters.

Features:
    - IV data binning and processing
    - Calculation of differential conductance and resistance
    - Sorting and mapping of IV data over the y-axis
    - Temperature evaluation for sweeps
    - Logging for debugging and tracking changes to evaluation parameters

Classes:
    - IVEvaluation: Handles IV data evaluation, including getting and processing IV sweeps.

Methods:
    - get_differentials(dictionary): Computes differential conductance and resistance.
    - getSingleIV(y_key, specific_trigger): Retrieves a single IV sweep with binning and temperature handling.
    - getMaps(y_bounds): Processes and maps IV sweeps over a voltage axis.
    - Property methods: Getters and setters for IV evaluation parameters such as voltage/current limits,
      binning settings, and upsampling options.

Author: Oliver Irtenkauf
Date: 2025-04-01
"""

import sys
import logging
import importlib

import numpy as np

from tqdm import tqdm
from scipy import constants
from h5py import File

from utilities.baseevaluation import BaseEvaluation
from utilities.basefunctions import linear_fit
from utilities.basefunctions import bin_y_over_x

logger = logging.getLogger(__name__)
importlib.reload(sys.modules["utilities.baseevaluation"])
importlib.reload(sys.modules["utilities.basefunctions"])


class IVEvaluation(BaseEvaluation):
    """
    Initializes an instance of the IVEvaluation class.

    Parameters
    ----------
    name : str, optional
        Name of the evaluation instance (default is "iv eva").
    """

    def __init__(
        self,
        name: str = "iv eva",
    ):
        """
        Description
        """
        super().__init__()
        self._iv_eva_name: str = name

        self.mapped = {
            "y_axis": np.array([]),
            "voltage_offset_1": np.array([]),
            "voltage_offset_2": np.array([]),
            "voltage_minimum": np.nan,
            "voltage_maximum": np.nan,
            "voltage_bins": np.nan,
            "voltage_axis": np.array([]),
            "current_minimum": np.nan,
            "current_maximum": np.nan,
            "current_bins": np.nan,
            "current_axis": np.array([]),
            "upsampling_current": 0,
            "upsampling_voltage": 0,
            "eva_current": True,
            "eva_voltage": True,
            "eva_temperature": True,
        }

        self.up_sweep = self.get_empty_dictionary()
        self.down_sweep = self.get_empty_dictionary()

        self.setV(2.0e-3, voltage_bins=100)
        self.setI(1.0e-6, current_bins=100)

        logger.info("(%s) ... IVEvaluation initialized.", self._iv_eva_name)

    def get_empty_dictionary(self, len_y: int = 0):
        logger.debug("(%s) get_empty_dictionary(len_y=%s)", self._iv_eva_name, len_y)
        """
        Returns a dictionary initialized with empty arrays for storing IV data.

        Parameters
        ----------
        len_y : int, optional
            Number of y-axis data points (default is 0).

        Returns
        -------
        dict
            A dictionary containing empty arrays for IV evaluation.
        """
        len_v = np.shape(self.voltage_axis)[0]
        len_i = np.shape(self.current_axis)[0]
        if not len_y:
            len_y = np.shape(self.y_unsorted)[0]
        return {
            "iv_tuples": [None] * len_y,
            "temperature": np.full(len_y, np.nan, dtype="float64"),
            "time_start": np.full(len_y, np.nan, dtype="float64"),
            "time_stop": np.full(len_y, np.nan, dtype="float64"),
            "current": np.full((len_y, len_v), np.nan, dtype="float64"),
            "voltage": np.full((len_y, len_i), np.nan, dtype="float64"),
            "time_current": np.full((len_y, len_v), np.nan, dtype="float64"),
            "time_voltage": np.full((len_y, len_i), np.nan, dtype="float64"),
            "temperature_current": np.full((len_y, len_v), np.nan, dtype="float64"),
            "temperature_voltage": np.full((len_y, len_i), np.nan, dtype="float64"),
            "differential_conductance": np.full(
                (len_y, len_v), np.nan, dtype="float64"
            ),
            "differential_resistance": np.full((len_y, len_i), np.nan, dtype="float64"),
        }

    def setV(
        self,
        voltage_absolute: float = np.nan,
        voltage_minimum: float = np.nan,
        voltage_maximum: float = np.nan,
        voltage_bins: int = 0,
    ):
        """
        Sets the voltage-axis.

        Parameters
        ----------
        voltage_absolute : float, optional
            If provided, sets voltage_minimum and voltage_maximum symmetrically.
        voltage_minimum : float, optional
            Minimum voltage value.
        voltage_maximum : float, optional
            Maximum voltage value.
        voltage_bins : float, optional
            Number of bins minus 1.
        """

        if not np.isnan(voltage_absolute):
            voltage_minimum = -voltage_absolute
            voltage_maximum = +voltage_absolute

        if not np.isnan(voltage_minimum):
            self.voltage_minimum = voltage_minimum
        if not np.isnan(voltage_maximum):
            self.voltage_maximum = voltage_maximum
        if voltage_bins:
            self.voltage_bins = voltage_bins

        # Calculate new voltage axis.
        self.voltage_axis = np.linspace(
            self.voltage_minimum,
            self.voltage_maximum,
            self.voltage_bins + 1,
        )

        logger.debug(
            "(%s) setV(%s, %s, %s)",
            self._iv_eva_name,
            self.voltage_minimum,
            self.voltage_maximum,
            self.voltage_bins,
        )

    def setI(
        self,
        current_absolute: float = np.nan,
        current_minimum: float = np.nan,
        current_maximum: float = np.nan,
        current_bins: int = 0,
    ):
        """
        Sets the current-axis.

        Parameters
        ----------
        current_absolute : float, optional
            If provided, sets current_minimum and current_maximum symmetrically.
        current_minimum : float, optional
            Minimum current value.
        current_maximum : float, optional
            Maximum current value.
        current_bins : float, optional
            Number of bins minus 1.
        """

        if not np.isnan(current_absolute):
            current_minimum = -current_absolute
            current_maximum = +current_absolute

        if not np.isnan(current_minimum):
            self.current_minimum = current_minimum
        if not np.isnan(current_maximum):
            self.current_maximum = current_maximum
        if current_bins:
            self.current_bins = current_bins

        # Calculate new current axis.
        self.current_axis = np.linspace(
            self.current_minimum,
            self.current_maximum,
            self.current_bins + 1,
        )

        logger.debug(
            "(%s) setI(%s, %s, %s)",
            self._iv_eva_name,
            self.current_minimum,
            self.current_maximum,
            self.current_bins,
        )

    def get_current_voltage(self, specific_key: str):
        """
        Retrieves and processes current and voltage data from an HDF5 file.

        Parameters
        ----------
        specific_key : str
            The specific measurement key used to locate the data within the file.

        Returns
        -------
        v_raw : np.ndarray
            The processed voltage values.
        i_raw : np.ndarray
            The processed current values.
        trigger : np.ndarray
            The trigger signals from the measurement.
        time : np.ndarray
            The corresponding time values.
        v1_off : float
            The offset for voltage channel V1.
        v2_off : float
            The offset for voltage channel V2.
        """
        logger.debug(
            "(%s) get_current_voltage(%s)",
            self._iv_eva_name,
            specific_key,
        )

        file_name = f"{self.file_directory}{self.file_folder}{self.file_name}"

        with File(file_name, "r") as data_file:
            # Retrieve Offset Dataset
            measurement_data_offset = np.array(
                data_file.get(
                    f"measurement/{self.measurement_key}/{specific_key}/offset/adwin"
                )
            )
            # Calculate Offsets
            v1_off = np.nanmean(np.array(measurement_data_offset["V1"]))
            v2_off = np.nanmean(np.array(measurement_data_offset["V2"]))

            # Retrieve Sweep Dataset
            measurement_data_sweep = np.array(
                data_file.get(
                    f"measurement/{self.measurement_key}/{specific_key}/sweep/adwin"
                )
            )

            # Get Voltage Readings of Adwin
            trigger = np.array(measurement_data_sweep["trigger"], dtype="int")
            time = np.array(measurement_data_sweep["time"], dtype="float64")
            v1 = np.array(measurement_data_sweep["V1"], dtype="float64")
            v2 = np.array(measurement_data_sweep["V2"], dtype="float64")

        # Calculate V, I
        v_raw = (v1 - v1_off) / self.voltage_amplification_1

        i_raw = (v2 - v2_off) / self.voltage_amplification_2 / self.reference_resistor

        return v_raw, i_raw, trigger, time, v1_off, v2_off

    def get_backup_temperatures(self):
        """
        Retrieves backup temperature data from status.

        Returns
        -------
        tuple of np.ndarray
            time : np.ndarray
                Array of time values.
            temperature : np.ndarray
                Array of corresponding temperature values.
        """
        file_name = f"{self.file_directory}{self.file_folder}{self.file_name}"
        with File(file_name, "r") as data_file:
            status = np.array(data_file.get("status/bluefors/temperature/MCBJ"))
            time = status["time"]
            temperature = status["T"]
            return time, temperature

    def check_for_temperatures(self, key: str):
        """
        Checks if temperature data from Bluefors is available for a given measurement key.

        Parameters
        ----------
        key : str
            The specific measurement key to check.

        Returns
        -------
        bool
            True if Bluefors temperature data is available, False otherwise.
        """
        file_name = f"{self.file_directory}{self.file_folder}{self.file_name}"
        with File(file_name, "r") as data_file:
            check = data_file.__contains__(
                f"measurement/{self.measurement_key}/{key}/sweep/bluefors"
            )
            return check

    def get_sweep_temperatures(self, key: str):
        """
        Retrieves temperature data from a specific sweep measurement.

        Parameters
        ----------
        key : str
            The specific measurement key to retrieve temperature data from.

        Returns
        -------
        tuple of np.ndarray
            time : np.ndarray
                Array of time values.
            temperature : np.ndarray
                Array of corresponding temperature values.
        """
        file_name = f"{self.file_directory}{self.file_folder}{self.file_name}"
        with File(file_name, "r") as data_file:
            bluefors = np.array(
                data_file.get(
                    f"measurement/{self.measurement_key}/{key}/sweep/bluefors"
                )
            )
            time = bluefors["time"]
            temperature = bluefors["Tsample"]
            return time, temperature

    def get_iv_binning_done(
        self,
        index: int,
        specific_trigger: int,
        dictionary: dict,
        v_raw: np.ndarray,
        i_raw: np.ndarray,
        trigger: np.ndarray,
        time: np.ndarray,
    ):
        """
        Bins IV data based on a specific trigger and stores the results in the given dictionary.

        Parameters
        ----------
        index : int
            Index in the dictionary to store the results.
        specific_trigger : int
            The trigger value to filter the data.
        dictionary : dict
            The dictionary to store the processed IV data.
        v_raw : np.ndarray
            Raw voltage data.
        i_raw : np.ndarray
            Raw current data.
        trigger : np.ndarray
            Array containing trigger values.
        time : np.ndarray
            Time data corresponding to the measurements.
        """
        logger.debug(
            "(%s) get_iv_binning_done(index=%d, specific_trigger=%d)",
            self._iv_eva_name,
            index,
            specific_trigger,
        )

        # Filter data based on the specific trigger
        time_filtered = time[trigger == specific_trigger].copy()
        v_raw_filtered = v_raw[trigger == specific_trigger].copy()
        i_raw_filtered = i_raw[trigger == specific_trigger].copy()

        # Ensure there is data available before accessing elements
        if (
            time_filtered.size == 0
            or v_raw_filtered.size == 0
            or i_raw_filtered.size == 0
        ):
            logger.warning(
                "(%s) No data found for trigger %d", self._iv_eva_name, specific_trigger
            )
            return

        # Store time range
        dictionary["time_start"][index] = time_filtered[0]
        dictionary["time_stop"][index] = time_filtered[-1]

        # Store raw IV data tuples
        dictionary["iv_tuples"][index] = [i_raw_filtered, v_raw_filtered]

        # Bin current data if enabled
        if self.eva_current:
            dictionary["current"][index, :], _ = bin_y_over_x(
                v_raw_filtered,
                i_raw_filtered,
                self.voltage_axis,
                upsampling=self.upsampling_current,
            )
            dictionary["time_current"][index, :], _ = bin_y_over_x(
                v_raw_filtered,
                time_filtered,
                self.voltage_axis,
                upsampling=self.upsampling_current,
            )

        # Bin voltage data if enabled
        if self.eva_voltage:
            dictionary["voltage"][index, :], _ = bin_y_over_x(
                i_raw_filtered,
                v_raw_filtered,
                self.current_axis,
                upsampling=self.upsampling_voltage,
            )
            dictionary["time_voltage"][index, :], _ = bin_y_over_x(
                i_raw_filtered,
                time_filtered,
                self.current_axis,
                upsampling=self.upsampling_voltage,
            )

    def get_temperature_binning_done(
        self,
        index: int,
        dictionary: dict,
        temporary_time: np.ndarray,
        temporary_temperature: np.ndarray,
    ):
        """
        Performs binning of temperature data along current and voltage axes.

        Parameters
        ----------
        index : int
            Index of the current dataset.
        dictionary : dict
            Dictionary containing data arrays for binning.
        temporary_time : np.ndarray
            Array of time values corresponding to temperature measurements.
        temporary_temperature : np.ndarray
            Array of temperature values to be binned.
        """

        logger.debug("(%s) get_temperature_binning_done(...)", self._iv_eva_name)

        if self.eva_current:
            # Fit a linear time axis for the current data
            temporary_time_current = linear_fit(dictionary["time_current"][index])

            # Ensure time is sorted in ascending order
            if temporary_time_current[0] > temporary_time_current[1]:
                temporary_time_current = np.flip(temporary_time_current)

            # Find indices where time values fall within the fitted range
            indices = np.where(
                np.logical_and(
                    temporary_time >= temporary_time_current[0],
                    temporary_time <= temporary_time_current[-1],
                )
            )

            # Bin temperature values along the current axis
            dictionary["temperature_current"][index, :], _ = bin_y_over_x(
                temporary_time[indices],
                temporary_temperature[indices],
                temporary_time_current,
                upsampling=1000,
            )

        if self.eva_voltage:
            # Fit a linear time axis for the voltage data
            temporary_time_voltage = linear_fit(dictionary["time_voltage"][index])

            # Ensure time is sorted in ascending order
            if temporary_time_voltage[0] > temporary_time_voltage[1]:
                temporary_time_voltage = np.flip(temporary_time_voltage)

            # Find indices where time values fall within the fitted range
            indices = np.where(
                np.logical_and(
                    temporary_time >= temporary_time_voltage[0],
                    temporary_time <= temporary_time_voltage[-1],
                )
            )

            # Bin temperature values along the voltage axis
            dictionary["temperature_voltage"][index, :], _ = bin_y_over_x(
                temporary_time[indices],
                temporary_temperature[indices],
                temporary_time_voltage,
                upsampling=1000,
            )

    def sort_mapped_over_y(self, y_bounds: list[int]):
        """
        Sorts the mapped data based on the y-axis values and applies optional bounds.

        Parameters
        ----------
        y_bounds : list[int]
            A list containing two integers defining the lower and upper bounds for selecting
            a subset of the sorted y-axis data.
        """
        logger.debug("(%s) sort_mapped_over_y(%s)", self._iv_eva_name, y_bounds)

        # Sort indices based on y_unsorted values
        indices = np.argsort(self.y_unsorted)

        # Apply sorting to y-related data
        self.y_axis = self.y_unsorted[indices]
        self.voltage_offset_1 = self.voltage_offset_1[indices]
        self.voltage_offset_2 = self.voltage_offset_2[indices]

        # Apply optional y-axis bounds if provided
        if y_bounds:
            lower, upper = y_bounds
            self.y_axis = self.y_axis[lower:upper]
            self.voltage_offset_1 = self.voltage_offset_1[lower:upper]
            self.voltage_offset_2 = self.voltage_offset_2[lower:upper]
        return indices

    def sort_dictionary_over_y(
        self,
        indices: np.ndarray,
        y_bounds: list[int],
        dictionary: dict,
    ):
        """
        Sorts the data in the dictionary based on the provided indices and applies optional
        y-axis bounds to the data.

        Parameters
        ----------
        indices : np.ndarray
            An array of indices to reorder the data in the dictionary.
        y_bounds : list[int]
            A list containing two integers defining the lower and upper bounds for selecting
            a subset of the sorted data.
        dictionary : dict
            The dictionary containing the data to be sorted. Keys include 'time_start', 'time_stop',
            'current', 'voltage', etc., with values to be reordered and optionally truncated by bounds.
        """
        logger.debug(
            "(%s) sort_dictionary_over_y(%s, %s, %s)",
            self._iv_eva_name,
            indices,
            y_bounds,
            dictionary,
        )

        # Sort time-related entries in the dictionary
        dictionary["time_start"] = dictionary["time_start"][indices]
        dictionary["time_stop"] = dictionary["time_stop"][indices]

        # Apply optional y-axis bounds for time
        if y_bounds:
            lower, upper = y_bounds
            dictionary["time_start"] = dictionary["time_start"][lower:upper]
            dictionary["time_stop"] = dictionary["time_stop"][lower:upper]

        # Sort current-related data if eva_current is True
        if self.eva_current:
            dictionary["current"] = dictionary["current"][indices, :]
            dictionary["time_current"] = dictionary["time_current"][indices, :]

            # Sort temperature-related data for current if eva_temperature is True
            if self.eva_temperature:
                dictionary["temperature_current"] = dictionary["temperature_current"][
                    indices, :
                ]

            # Apply y-axis bounds for current and time_current
            if y_bounds:
                lower, upper = y_bounds
                dictionary["current"] = dictionary["current"][lower:upper, :]
                dictionary["time_current"] = dictionary["time_current"][lower:upper, :]

                # Apply bounds for temperature_current if eva_temperature is True
                if self.eva_temperature:
                    dictionary["temperature_current"] = dictionary[
                        "temperature_current"
                    ][lower:upper, :]

        # Sort voltage-related data if eva_voltage is True
        if self.eva_voltage:
            dictionary["voltage"] = dictionary["voltage"][indices, :]
            dictionary["time_voltage"] = dictionary["time_voltage"][indices, :]

            # Sort temperature-related data for voltage if eva_temperature is True
            if self.eva_temperature:
                dictionary["temperature_voltage"] = dictionary["temperature_voltage"][
                    indices, :
                ]

            # Apply y-axis bounds for voltage and time_voltage
            if y_bounds:
                lower, upper = y_bounds
                dictionary["voltage"] = dictionary["voltage"][lower:upper, :]
                dictionary["time_voltage"] = dictionary["time_voltage"][lower:upper, :]

                # Apply bounds for temperature_voltage if eva_temperature is True
                if self.eva_temperature:
                    dictionary["temperature_voltage"] = dictionary[
                        "temperature_voltage"
                    ][lower:upper, :]

    def get_differentials(self, dictionary: dict):
        """
        Calculates the differentials for conductance and resistance based on the provided dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary containing data for 'current', 'voltage', 'temperature_current', and 'temperature_voltage'.
            The function updates the dictionary with 'differential_conductance', 'differential_resistance',
            and 'temperature' based on the conditions.

        Updates the dictionary with the following:
        - 'differential_conductance': The differential conductance, calculated from the current and voltage data.
        - 'differential_resistance': The differential resistance, calculated from the voltage and current data.
        - 'temperature': The mean temperature, based on either the 'temperature_current' or 'temperature_voltage'.
        """
        logger.debug("(%s) get_differentials(%s)", self._iv_eva_name, dictionary)

        # Physical constant: conductance quantum (in SI units)
        conductance_quantum = constants.physical_constants["conductance quantum"][0]

        # Calculate differential conductance if eva_current is enabled
        if self.eva_current:
            # Compute the gradient of the current with respect to the voltage axis and divide by the conductance quantum
            dictionary["differential_conductance"] = (
                np.gradient(dictionary["current"], self.voltage_axis, axis=1)
                / conductance_quantum
            )
            # Compute the average temperature from the 'temperature_current' data
            dictionary["temperature"] = np.nanmean(
                dictionary["temperature_current"], axis=1
            )

        # Calculate differential resistance if eva_voltage is enabled
        if self.eva_voltage:
            # Compute the gradient of the voltage with respect to the current axis
            dictionary["differential_resistance"] = np.gradient(
                dictionary["voltage"], self.current_axis, axis=1
            )
            # Compute the average temperature from the 'temperature_voltage' data
            dictionary["temperature"] = np.nanmean(
                dictionary["temperature_voltage"], axis=1
            )

    def getSingleIV(self, y_key: str, specific_trigger: int = 1):
        """
        Retrieves a single IV curve and associated data, including voltage, current,
        temperature, and differential values.

        Parameters
        ----------
        y_key : str
            The key used to access specific measurement data for voltage, current, and temperature.

        specific_trigger : int, optional, default=1
            The trigger value used to filter the data for the IV curve.

        Returns
        -------
        dict
            A dictionary containing the binned data for voltage, current, temperature,
            and the differential conductance/resistance.
        """
        logger.debug(
            "(%s) getSingleIV(%s, %s)", self._iv_eva_name, y_key, specific_trigger
        )

        # Initialize an empty dictionary for a single IV curve
        single_iv = self.get_empty_dictionary(len_y=1)

        # Retrieve raw voltage, current, trigger, and time data
        (
            v_raw,
            i_raw,
            trigger,
            time,
            single_iv["voltage_offset_1"],
            single_iv["voltage_offset_2"],
        ) = self.get_current_voltage(y_key)

        # Bin the IV data
        self.get_iv_binning_done(
            0,  # Index for the first data point
            specific_trigger,  # Specific trigger to filter the data
            single_iv,  # Dictionary to store the binned data
            v_raw,  # Raw voltage data
            i_raw,  # Raw current data
            trigger,  # Trigger data
            time,  # Time data
        )

        # If temperature data is needed, check for availability and bin the temperature data
        if self.eva_temperature:
            if self.check_for_temperatures(y_key):
                # If temperature data is available, get the sweep temperatures
                time, temperature = self.get_sweep_temperatures(y_key)
            else:
                # Otherwise, use the backup temperature data
                time, temperature = self.get_backup_temperatures()

            # Bin the temperature data
            self.get_temperature_binning_done(
                0,  # Index for the first data point
                single_iv,  # Dictionary to store the binned temperature data
                time,  # Time data
                temperature,  # Temperature data
            )

        # Calculate the differentials (conductance, resistance)
        self.get_differentials(single_iv)

        # Return the dictionary with the binned and calculated data
        return single_iv

    def getMaps(self, y_bounds: list[int] = []):
        """
        getMaps()
        - Calculates current (I) and voltage (V), and splits data for up and down sweeps.
        - Maps I, differential conductance, and temperature over a linear voltage axis.
        - Saves the start and stop times as well as offsets.
        - Sorts by the y-axis.

        Parameters
        ----------
        y_bounds : list of int, optional
            The bounds for the y-axis to filter the data. If not specified, the entire range is used.
        """
        logger.info("(%s) getMaps()", self._iv_eva_name)

        # Initialize voltage offset values
        len_y = np.shape(self.y_unsorted)[0]
        self.voltage_offset_1 = np.full(len_y, np.nan, dtype="float64")
        self.voltage_offset_2 = np.full(len_y, np.nan, dtype="float64")

        # Initialize sweep dictionaries for up and down triggers, if specified
        if self.index_trigger_up is not None:
            self.up_sweep = self.get_empty_dictionary()
        if self.index_trigger_down is not None:
            self.down_sweep = self.get_empty_dictionary()

        # Retrieve single IV curves for both up and down sweeps if y_0_key is defined
        if self.y_0_key != "":
            if self.index_trigger_up is not None:
                self.up_sweep["plain"] = self.getSingleIV(
                    self.y_0_key, self.index_trigger_up
                )
            if self.index_trigger_down is not None:
                self.down_sweep["plain"] = self.getSingleIV(
                    self.y_0_key, self.index_trigger_down
                )

        # Check for the availability of temperature data
        check = self.check_for_temperatures(self.specific_keys[0])
        if not check:
            # If no temperature data, fallback to backup temperature data
            temporary_time, temporary_temperature = self.get_backup_temperatures()

        # Iterate over the keys for each measurement
        for index, key in enumerate(tqdm(self.specific_keys)):
            # Get I, V from ADwin data for the current key
            (
                v_raw,
                i_raw,
                trigger,
                time,
                self.voltage_offset_1[index],
                self.voltage_offset_2[index],
            ) = self.get_current_voltage(key)

            # Process binning for up and down sweeps if applicable
            if self.index_trigger_up is not None:
                self.get_iv_binning_done(
                    index,
                    self.index_trigger_up,
                    self.up_sweep,
                    v_raw,
                    i_raw,
                    trigger,
                    time,
                )

            if self.index_trigger_down is not None:
                self.get_iv_binning_done(
                    index,
                    self.index_trigger_down,
                    self.down_sweep,
                    v_raw,
                    i_raw,
                    trigger,
                    time,
                )

            # If temperature data is enabled, process temperature binning
            if self.eva_temperature:
                if check:
                    # Retrieve sweep temperatures if available
                    temporary_time, temporary_temperature = self.get_sweep_temperatures(
                        key
                    )

                # Bin temperature data for both up and down sweeps
                if self.index_trigger_up is not None:
                    self.get_temperature_binning_done(
                        index,
                        self.up_sweep,
                        temporary_time,
                        temporary_temperature,
                    )

                if self.index_trigger_down is not None:
                    self.get_temperature_binning_done(
                        index,
                        self.down_sweep,
                        temporary_time,
                        temporary_temperature,
                    )

        # Sort the data by the y-axis using the specified bounds
        indices = self.sort_mapped_over_y(y_bounds)

        # Apply sorting to the up and down sweep data if applicable
        if self.index_trigger_up is not None:
            self.sort_dictionary_over_y(indices, y_bounds, self.up_sweep)

        if self.index_trigger_down is not None:
            self.sort_dictionary_over_y(indices, y_bounds, self.down_sweep)

        # Calculate differential conductance and resistance for both sweeps
        if self.index_trigger_up is not None:
            self.get_differentials(self.up_sweep)
        if self.index_trigger_down is not None:
            self.get_differentials(self.down_sweep)

    ### IV Properties ###

    @property
    def eva_current(self):
        """Get the value of eva_current."""
        return self.mapped["eva_current"]

    @eva_current.setter
    def eva_current(self, eva_current):
        """Set the value of eva_current."""
        self.mapped["eva_current"] = eva_current
        logger.debug("(%s) eva_current = %s", self._iv_eva_name, eva_current)

    @property
    def eva_voltage(self):
        """Get the value of eva_voltage."""
        return self.mapped["eva_voltage"]

    @eva_voltage.setter
    def eva_voltage(self, eva_voltage):
        """Set the value of eva_voltage."""
        self.mapped["eva_voltage"] = eva_voltage
        logger.debug("(%s) eva_voltage = %s", self._iv_eva_name, eva_voltage)

    @property
    def eva_temperature(self):
        """Get the value of eva_temperature."""
        return self.mapped["eva_temperature"]

    @eva_temperature.setter
    def eva_temperature(self, eva_temperature):
        """Set the value of eva_temperature."""
        self.mapped["eva_temperature"] = eva_temperature
        logger.debug("(%s) eva_temperature = %s", self._iv_eva_name, eva_temperature)

    @property
    def voltage_offset_1(self):
        """Get the value of voltage_offset_1."""
        return self.mapped["voltage_offset_1"]

    @voltage_offset_1.setter
    def voltage_offset_1(self, voltage_offset_1):
        """Set the value of voltage_offset_1."""
        self.mapped["voltage_offset_1"] = voltage_offset_1

    @property
    def voltage_offset_2(self):
        """Get the value of voltage_offset_2."""
        return self.mapped["voltage_offset_2"]

    @voltage_offset_2.setter
    def voltage_offset_2(self, voltage_offset_2):
        """Set the value of voltage_offset_2."""
        self.mapped["voltage_offset_2"] = voltage_offset_2

    @property
    def y_axis(self):
        """Get the value of y_axis."""
        return self.mapped["y_axis"]

    @y_axis.setter
    def y_axis(self, y_axis):
        """Set the value of y_axis."""
        self.mapped["y_axis"] = y_axis

    @property
    def voltage_minimum(self):
        """Get the value of voltage_minimum."""
        return self.mapped["voltage_minimum"]

    @voltage_minimum.setter
    def voltage_minimum(self, voltage_minimum):
        """Set the value of voltage_minimum."""
        self.mapped["voltage_minimum"] = voltage_minimum

    @property
    def voltage_maximum(self):
        """Get the value of voltage_maximum."""
        return self.mapped["voltage_maximum"]

    @voltage_maximum.setter
    def voltage_maximum(self, voltage_maximum):
        """Set the value of voltage_maximum."""
        self.mapped["voltage_maximum"] = voltage_maximum

    @property
    def voltage_bins(self):
        """Get the value of voltage_bins."""
        return self.mapped["voltage_bins"]

    @voltage_bins.setter
    def voltage_bins(self, voltage_bins):
        """Set the value of voltage_bins."""
        self.mapped["voltage_bins"] = voltage_bins

    @property
    def voltage_axis(self):
        """Get the value of voltage_axis."""
        return self.mapped["voltage_axis"]

    @voltage_axis.setter
    def voltage_axis(self, voltage_axis):
        """Set the value of voltage_axis."""
        self.mapped["voltage_axis"] = voltage_axis

    @property
    def current_minimum(self):
        """Get the value of current_minimum."""
        return self.mapped["current_minimum"]

    @current_minimum.setter
    def current_minimum(self, current_minimum):
        """Set the value of current_minimum."""
        self.mapped["current_minimum"] = current_minimum

    @property
    def current_maximum(self):
        """Get the value of current_maximum."""
        return self.mapped["current_maximum"]

    @current_maximum.setter
    def current_maximum(self, current_maximum):
        """Set the value of current_maximum."""
        self.mapped["current_maximum"] = current_maximum

    @property
    def current_bins(self):
        """Get the value of current_bins."""
        return self.mapped["current_bins"]

    @current_bins.setter
    def current_bins(self, current_bins):
        """Set the value of current_bins."""
        self.mapped["current_bins"] = current_bins

    @property
    def current_axis(self):
        """Get the value of current_axis."""
        return self.mapped["current_axis"]

    @current_axis.setter
    def current_axis(self, current_axis):
        """Set the value of current_axis."""
        self.mapped["current_axis"] = current_axis

    @property
    def upsampling_current(self):
        """Get the value of upsampling_current."""
        return self.mapped["upsampling_current"]

    @upsampling_current.setter
    def upsampling_current(self, upsampling_current: int):
        """Set the value of upsampling_current."""
        self.mapped["upsampling_current"] = upsampling_current

    @property
    def upsampling_voltage(self):
        """Get the value of upsampling_voltage."""
        return self.mapped["upsampling_voltage"]

    @upsampling_voltage.setter
    def upsampling_voltage(self, upsampling_voltage: int):
        """Set the value of upsampling_voltage."""
        self.mapped["upsampling_voltage"] = upsampling_voltage
