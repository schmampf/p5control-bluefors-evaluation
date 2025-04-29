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
      binning settings, and upsample options.

Author: Oliver Irtenkauf
Date: 2025-04-01
"""

# region imports

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
from utilities.basefunctions import bin_z_over_y
from utilities.basefunctions import get_amplitude
from utilities.basefunctions import make_even_spaced
from utilities.basefunctions import downsample_signals_by_time

logger = logging.getLogger(__name__)
importlib.reload(sys.modules["utilities.baseevaluation"])
importlib.reload(sys.modules["utilities.basefunctions"])

# endregion


class IVEvaluation(BaseEvaluation):

    # region __init__, empf_dict, setIVAT
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
        # self._iv_eva_name: str = name
        # super().__init__()

        # self.mapped = {
        # }

        self.index_trigger: int = 1
        self.evaluated = self.get_empty_dictionary()

        self.up_sweep = self.get_empty_dictionary()
        self.down_sweep = self.get_empty_dictionary()

        # self.setV(2.0e-3, voltage_bins=100)
        # self.setI(1.0e-6, current_bins=100)
        # self.setA(0, 1, 500)
        # self.setT(0, 1.5, 500)

        # logger.info("(%s) ... IVEvaluation initialized.", self._iv_eva_name)

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
            "iv_tuples_raw": [None] * len_y,
            "iv_tuples": [None] * len_y,
            "downsample_frequency": np.full(len_y, np.nan, dtype="float64"),
            "temperature": np.full(len_y, np.nan, dtype="float64"),
            "time_start": np.full(len_y, np.nan, dtype="float64"),
            "time_stop": np.full(len_y, np.nan, dtype="float64"),
            "current": np.full((len_y, len_v), np.nan, dtype="float64"),
            "voltage": np.full((len_y, len_i), np.nan, dtype="float64"),
            "time_current": np.full((len_y, len_v), np.nan, dtype="float64"),
            "time_voltage": np.full((len_y, len_i), np.nan, dtype="float64"),
            # "temperature_current": np.full((len_y, len_v), np.nan, dtype="float64"),
            # "temperature_voltage": np.full((len_y, len_i), np.nan, dtype="float64"),
            # "differential_conductance": np.full(
            #     (len_y, len_v), np.nan, dtype="float64"
            # ),
            # "differential_resistance": np.full((len_y, len_i), np.nan, dtype="float64"),
        }

    # endregion

    # region internal functions
    def get_current_voltage(self, specific_key: str = ""):
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
            if specific_key != 0:
                measurement_data_offset = np.array(
                    data_file.get(
                        f"measurement/{self.measurement_key}/{specific_key}/offset/adwin"
                    )
                )
            else:
                measurement_data_offset = np.array(
                    data_file.get(f"measurement/{self.measurement_key}/offset/adwin")
                )

            # Calculate Offsets
            v1_off = np.nanmean(np.array(measurement_data_offset["V1"]))
            v2_off = np.nanmean(np.array(measurement_data_offset["V2"]))

            # Retrieve Sweep Dataset
            if specific_key != 0:
                measurement_data_sweep = np.array(
                    data_file.get(
                        f"measurement/{self.measurement_key}/{specific_key}/sweep/adwin"
                    )
                )
            else:
                measurement_data_sweep = np.array(
                    data_file.get(f"measurement/{self.measurement_key}/sweep/adwin")
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
        dictionary["iv_tuples_raw"][index] = [
            i_raw_filtered,
            v_raw_filtered,
            time_filtered,
        ]

        # Downsample signals
        (
            i_raw_downsampled,
            v_raw_downsampled,
            t_raw_downsampled,
            downsample_counts,
            downsample_frequency,
        ) = downsample_signals_by_time(
            current=i_raw_filtered,
            voltage=v_raw_filtered,
            time=time_filtered,
            target_frequency=self.downsample_frequency,
            savety_factor=3,
            prefer_prime=True,
            power_threshold=0.95,
        )

        # Store raw IV data tuples
        dictionary["iv_tuples"][index] = [
            i_raw_downsampled,
            v_raw_downsampled,
            t_raw_downsampled,
            downsample_counts,
        ]
        dictionary["downsample_frequency"][index] = downsample_frequency

        # Bin current data if enabled
        if self.eva_current:
            dictionary["current"][index, :], _ = bin_y_over_x(
                v_raw_downsampled,
                i_raw_downsampled,
                self.voltage_axis,
                upsample=self.upsample_current,
            )
            dictionary["time_current"][index, :], _ = bin_y_over_x(
                v_raw_downsampled,
                t_raw_downsampled,
                self.voltage_axis,
                upsample=self.upsample_current,
            )

        # Bin voltage data if enabled
        if self.eva_voltage:
            dictionary["voltage"][index, :], _ = bin_y_over_x(
                i_raw_downsampled,
                v_raw_downsampled,
                self.current_axis,
                upsample=self.upsample_voltage,
            )
            dictionary["time_voltage"][index, :], _ = bin_y_over_x(
                i_raw_downsampled,
                t_raw_downsampled,
                self.current_axis,
                upsample=self.upsample_voltage,
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
                upsample=1000,
                upsample_method="nearest",
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
                upsample=1000,
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

    def getMapsEvenSpaced(self, already_evaluated: list[dict]):
        logger.info("(%s) getMapsAmplitude()", self._iv_eva_name)
        evaluated = []
        temp_yaxis = np.copy(self.y_axis)
        self.y_axis = make_even_spaced(self.y_axis)
        for index_to_evaluate, to_evaluate in enumerate(already_evaluated):
            evaluated.append(self.get_empty_dictionary())
            for string in [
                "current",
                "time_current",
                "temperature_current",
                "voltage",
                "time_voltage",
                "temperature_voltage",
                "differential_conductance",
                "differential_resistance",
            ]:
                (
                    evaluated[index_to_evaluate][string],
                    evaluated[index_to_evaluate][f"{string}_counter"],
                ) = bin_z_over_y(
                    temp_yaxis,
                    to_evaluate[string],
                    self.y_axis,
                )
            for string in [
                "temperature",
                "time_start",
                "time_stop",
            ]:

                (
                    evaluated[index_to_evaluate][string],
                    evaluated[index_to_evaluate][f"{string}_counter"],
                ) = bin_y_over_x(
                    temp_yaxis,
                    to_evaluate[string],
                    self.y_axis,
                )
        return tuple(evaluated)

    # endregion

    # region main functions
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

    def getMaps(self, trigger_indices: list[int] = [1], y_bounds: list[int] = []):
        """
        Processes IV measurement data and returns mapped quantities over a linear voltage axis.

        This function:
        - Retrieves raw I(V) data for each measurement key.
        - Splits sweeps by trigger indices (e.g., for up/down sweep separation).
        - Computes differential conductance and optional temperature mapping.
        - Applies voltage offset correction and y-axis sorting/filtering.
        - Returns all results as dictionaries, one per trigger index.

        Parameters
        ----------
        trigger_indices : list of int, optional
            Trigger indices to separate multiple sweeps or events. Default is [1].

        y_bounds : list of int, optional
            Bounds for filtering the data along the y-axis. If empty, the full range is used.

        Returns
        -------
        tuple of dict
            Each dictionary contains mapped results (I, V, dI/dV, temperature, etc.)
            for one trigger index.
        """
        logger.info("(%s) getMaps()", self._iv_eva_name)

        # Allocate voltage offset arrays, one per y-position
        len_y = np.shape(self.y_unsorted)[0]
        self.voltage_offset_1 = np.full(len_y, np.nan, dtype="float64")
        self.voltage_offset_2 = np.full(len_y, np.nan, dtype="float64")

        # Precompute sorting indices along the y-axis
        sort_indices = self.sort_mapped_over_y(y_bounds)

        # Determine whether temperature data is available
        has_temp_data = self.check_for_temperatures(self.specific_keys[0])
        if not has_temp_data:
            # Fallback to backup temperature data
            temporary_time, temporary_temperature = self.get_backup_temperatures()

        evaluated = []
        for i_trigger_index, trigger_index in enumerate(trigger_indices):
            # Create an empty results dictionary for this trigger
            evaluated.append(self.get_empty_dictionary())

            # If a fixed y-value key is defined, retrieve its IV data
            if self.y_0_key != "":
                evaluated[i_trigger_index]["plain"] = self.getSingleIV(
                    self.y_0_key,
                    trigger_index,
                )
                self.downsample_frequency = float(
                    evaluated[i_trigger_index]["plain"]["downsample_frequency"]
                )

            # Iterate over the keys for each measurement
            for index, key in enumerate(tqdm(self.specific_keys)):
                # Extract raw voltage, current, trigger, and time data
                (
                    v_raw,
                    i_raw,
                    trigger,
                    time,
                    self.voltage_offset_1[index],
                    self.voltage_offset_2[index],
                ) = self.get_current_voltage(key)

                # Bin current and voltage data
                self.get_iv_binning_done(
                    index,
                    trigger_index,
                    evaluated[i_trigger_index],
                    v_raw,
                    i_raw,
                    trigger,
                    time,
                )

                # If temperature evaluation is enabled
                if self.eva_temperature:
                    if has_temp_data:
                        # Get temperature data for this key
                        temporary_time, temporary_temperature = (
                            self.get_sweep_temperatures(key)
                        )

                    # Bin temperature data
                    self.get_temperature_binning_done(
                        index,
                        evaluated[i_trigger_index],
                        temporary_time,
                        temporary_temperature,
                    )

            # Apply sorting to the data based on y-axis order
            self.sort_dictionary_over_y(
                sort_indices, y_bounds, evaluated[i_trigger_index]
            )

            # Calculate differential conductance and resistance
            self.get_differentials(evaluated[i_trigger_index])

            if self.eva_even_spaced:
                return self.getMapsEvenSpaced(evaluated)
            else:
                return tuple(evaluated)

    def getMapsAmplitude(self, already_evaluated: list[dict]):
        logger.info("(%s) getMapsAmplitude()", self._iv_eva_name)
        evaluated = []
        amplitude = get_amplitude(self.mapped["y_axis"])
        for index_to_evaluate, to_evaluate in enumerate(already_evaluated):
            evaluated.append(self.get_empty_dictionary())
            for string in [
                "current",
                "time_current",
                "temperature_current",
                "voltage",
                "time_voltage",
                "temperature_voltage",
                "differential_conductance",
                "differential_resistance",
            ]:
                (
                    evaluated[index_to_evaluate][string],
                    evaluated[index_to_evaluate][f"{string}_counter"],
                ) = bin_z_over_y(
                    amplitude,
                    to_evaluate[string],
                    self.amplitude_axis,
                )
            for string in [
                "temperature",
                "time_start",
                "time_stop",
            ]:

                (
                    evaluated[index_to_evaluate][string],
                    evaluated[index_to_evaluate][f"{string}_counter"],
                ) = bin_y_over_x(
                    amplitude,
                    to_evaluate[string],
                    self.amplitude_axis,
                )

        return tuple(evaluated)

    def getMapsTemperature(self, already_evaluated: list[dict]):
        logger.info("(%s) getMapsTemperature()", self._iv_eva_name)
        evaluated = []
        for index_to_evaluate, to_evaluate in enumerate(already_evaluated):
            temperature = already_evaluated[index_to_evaluate]["temperature"]
            evaluated.append(self.get_empty_dictionary())
            for string in [
                "current",
                "time_current",
                "temperature_current",
                "differential_conductance",
                "voltage",
                "time_voltage",
                "temperature_voltage",
                "differential_resistance",
            ]:
                (
                    evaluated[index_to_evaluate][string],
                    evaluated[index_to_evaluate][f"{string}_counter"],
                ) = bin_z_over_y(
                    temperature,
                    to_evaluate[string],
                    self.temperature_axis,
                )
            for string in ["time_start", "time_stop", "temperature"]:

                (
                    evaluated[index_to_evaluate][string],
                    evaluated[index_to_evaluate][f"{string}_counter"],
                ) = bin_y_over_x(
                    temperature,
                    to_evaluate[string],
                    self.temperature_axis,
                )

        return tuple(evaluated)

    # endregion
