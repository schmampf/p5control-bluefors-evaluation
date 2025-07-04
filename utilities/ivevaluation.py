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

import os
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
        """
        Description
        """
        self._iv_eva_name: str = name
        BaseEvaluation.__init__(self)

        self.mapped = {
            "y_axis": np.array([]),
            "voltage_offset_1": np.array([]),
            "voltage_offset_2": np.array([]),
            "downsample_frequency": 3,
            "voltage_minimum": np.nan,
            "voltage_maximum": np.nan,
            "voltage_bins": np.nan,
            "voltage_axis": np.array([]),
            "current_minimum": np.nan,
            "current_maximum": np.nan,
            "current_bins": np.nan,
            "current_axis": np.array([]),
            "amplitude_minimum": np.nan,
            "amplitude_maximum": np.nan,
            "amplitude_bins": np.nan,
            "amplitude_axis": np.array([]),
            "temperature_minimum": np.nan,
            "temperature_maximum": np.nan,
            "temperature_bins": np.nan,
            "temperature_axis": np.array([]),
            "upsample_current": 137,
            "upsample_voltage": 137,
            "upsample_amplitude": 0,
            "upsample_temperature": 0,
            "eva_current": True,
            "eva_voltage": True,
            "eva_temperature": True,
            "eva_even_spaced": False,
        }

        self.temperature_t: np.ndarray | None = None
        self.temperature_T: np.ndarray | None = None

        self.index_trigger: int = 1
        self.evaluated = self.get_empty_dictionary()

        self.up_sweep = self.get_empty_dictionary()
        self.down_sweep = self.get_empty_dictionary()

        self.setV(2.0e-3, voltage_bins=100)
        self.setI(1.0e-6, current_bins=100)
        self.setA(0, 1, 500)
        self.setT(0, 1.5, 500)

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

    def setA(
        self,
        amplitude_minimum: float = np.nan,
        amplitude_maximum: float = np.nan,
        amplitude_bins: int = 0,
    ):
        """
        Sets the amplitude-axis.

        Parameters
        ----------
        amplitude_minimum : float, optional
            Minimum amplitude value.
        amplitude_maximum : float, optional
            Maximum amplitude value.
        amplitude_bins : float, optional
            Number of bins minus 1.
        """

        if not np.isnan(amplitude_minimum):
            self.amplitude_minimum = amplitude_minimum
        if not np.isnan(amplitude_maximum):
            self.amplitude_maximum = amplitude_maximum
        if amplitude_bins:
            self.amplitude_bins = amplitude_bins

        # Calculate new amplitude axis.
        self.amplitude_axis = np.linspace(
            self.amplitude_minimum,
            self.amplitude_maximum,
            self.amplitude_bins + 1,
        )

        logger.debug(
            "(%s) setA(%s, %s, %s)",
            self._iv_eva_name,
            self.amplitude_minimum,
            self.amplitude_maximum,
            self.amplitude_bins,
        )

    def setT(
        self,
        temperature_minimum: float = np.nan,
        temperature_maximum: float = np.nan,
        temperature_bins: int = 0,
    ):
        """
        Sets the temperature-axis.

        Parameters
        ----------
        temperature_minimum : float, optional
            Minimum temperature value.
        temperature_maximum : float, optional
            Maximum temperature value.
        temperature_bins : float, optional
            Number of bins minus 1.
        """

        if not np.isnan(temperature_minimum):
            self.temperature_minimum = temperature_minimum
        if not np.isnan(temperature_maximum):
            self.temperature_maximum = temperature_maximum
        if temperature_bins:
            self.temperature_bins = temperature_bins

        # Calculate new temperature axis.
        self.temperature_axis = np.linspace(
            self.temperature_minimum,
            self.temperature_maximum,
            self.temperature_bins + 1,
        )

        logger.debug(
            "(%s) setT(%s, %s, %s)",
            self._iv_eva_name,
            self.temperature_minimum,
            self.temperature_maximum,
            self.temperature_bins,
        )

    # endregion

    # region internal functions

    def getBackupTemperature(self):
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
        logger.info("(%s) getBackupTemperature()", self._iv_eva_name)
        file_name = os.path.join(
            self.file_directory,
            self.file_folder,
            self.file_name,
        )
        with File(file_name, "r") as data_file:
            status = np.array(data_file.get("status/bluefors/temperature/MCBJ"))
            self.temperature_t = status["time"]
            self.temperature_T = status["T"]

    def get_amplification(self, time: np.ndarray) -> tuple[int, int]:
        """
        retrieve the amplitudes amp1 and amp2 from self.amp_time, self.voltage_amplification_1, self.voltage_amplification_2
        """
        logger.debug("(%s) get_amplification(...)", self._iv_eva_name)

        amp_1 = bin_y_over_x(self.amp_t, self.amp_1, time)
        amp_2 = bin_y_over_x(self.amp_t, self.amp_2, time)
        # find the value that appears most often in the array
        amp_1 = int(np.nanmean(amp_1))
        amp_2 = int(np.nanmean(amp_2))

        return amp_1, amp_2

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
        file_name = os.path.join(
            self.file_directory,
            self.file_folder,
            self.file_name,
        )

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

        # get amps
        if self.voltage_amplification_1 is None or self.voltage_amplification_2 is None:
            amp_1, amp_2 = self.get_amplification(time)
        if self.voltage_amplification_1 is not None:
            amp_1 = self.voltage_amplification_1
        if self.voltage_amplification_2 is not None:
            amp_2 = self.voltage_amplification_2

        # Calculate V, I
        v_raw = (v1 - v1_off) / amp_1
        i_raw = (v2 - v2_off) / amp_2 / self.reference_resistor

        return v_raw, i_raw, trigger, time, v1_off, v2_off

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
        file_name = os.path.join(
            self.file_directory,
            self.file_folder,
            self.file_name,
        )
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
        file_name = os.path.join(
            self.file_directory,
            self.file_folder,
            self.file_name,
        )
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
        temporary_time: np.ndarray | None,
        temporary_temperature: np.ndarray | None,
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

        if temporary_time is None or temporary_temperature is None:
            raise ValueError("Temporary time and temperature arrays must not be None.")

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
                upsample_method="nearest",
            )

    def sort_y_axis(
        self,
        y_bounds: tuple[int, int] | None = None,
        y_lim: tuple[float, float] | None = None,
    ):
        """
        Sorts the mapped data based on the y-axis values and applies optional bounds.

        Parameters
        ----------
        y_bounds : list[int]
            A list containing two integers defining the lower and upper bounds for selecting
            a subset of the sorted y-axis data.
        """
        logger.debug("(%s) sort_by_y(%s)", self._iv_eva_name, y_bounds)

        # Sort indices based on y_unsorted values
        indices = np.argsort(self.y_unsorted)

        # Apply sorting to y-related data
        self.y_unsorted = self.y_unsorted[indices]
        self.specific_keys = [self.specific_keys[i] for i in indices]

        # Apply optional y-axis limits if provided
        if y_lim is not None:
            y_bounds = (
                int(np.argmin(np.abs(self.y_unsorted - y_lim[0]))),
                int(np.argmin(np.abs(self.y_unsorted - y_lim[1]))),
            )

        # Apply optional y-axis bounds if provided
        if y_bounds is not None:
            self.y_unsorted = self.y_unsorted[y_bounds[0] : y_bounds[1]]
            self.specific_keys = self.specific_keys[y_bounds[0] : y_bounds[1]]

        self.y_axis = self.y_unsorted

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
                if self.temperature_t is None:
                    self.getBackupTemperature()
                time, temperature = self.temperature_t, self.temperature_T

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

    def getMaps(
        self,
        trigger_indices: list[int,] = [1],
        y_bounds: tuple[int, int] | None = None,
        y_lim: tuple[float, float] | None = None,
    ) -> tuple[dict, ...]:
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
        logger.debug("(%s) getMaps()", self._iv_eva_name)

        # Allocate voltage offset arrays, one per y-position
        len_y = np.shape(self.y_unsorted)[0]
        self.voltage_offset_1 = np.full(len_y, np.nan, dtype="float64")
        self.voltage_offset_2 = np.full(len_y, np.nan, dtype="float64")

        # Precompute sorting indices along the y-axis
        self.sort_y_axis(y_bounds, y_lim)

        # Determine whether temperature data is available
        check_sweep_temperature = self.check_for_temperatures(self.specific_keys[0])
        if not check_sweep_temperature:
            if self.temperature_t is None:
                self.getBackupTemperature()

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
                    if check_sweep_temperature:
                        # Get temperature data for this key
                        temp_t, temp_T = self.get_sweep_temperatures(key)
                    else:
                        # Get Backup temperature data
                        temp_t, temp_T = self.temperature_t, self.temperature_T

                    # Bin temperature data
                    self.get_temperature_binning_done(
                        index,
                        evaluated[i_trigger_index],
                        temp_t,
                        temp_T,
                    )

            # Calculate differential conductance and resistance
            self.get_differentials(evaluated[i_trigger_index])

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

    # region iv properties

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
    def eva_even_spaced(self):
        """Get the value of eva_even_spaced."""
        return self.mapped["eva_even_spaced"]

    @eva_even_spaced.setter
    def eva_even_spaced(self, eva_even_spaced):
        """Set the value of eva_even_spaced."""
        self.mapped["eva_even_spaced"] = eva_even_spaced
        logger.debug("(%s) eva_even_spaced = %s", self._iv_eva_name, eva_even_spaced)

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
    def downsample_frequency(self):
        """Get the value of downsample_frequency."""
        return self.mapped["downsample_frequency"]

    @downsample_frequency.setter
    def downsample_frequency(self, downsample_frequency):
        """Set the value of downsample_frequency."""
        self.mapped["downsample_frequency"] = downsample_frequency

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
    def amplitude_minimum(self):
        """Get the value of amplitude_minimum."""
        return self.mapped["amplitude_minimum"]

    @amplitude_minimum.setter
    def amplitude_minimum(self, amplitude_minimum):
        """Set the value of amplitude_minimum."""
        self.mapped["amplitude_minimum"] = amplitude_minimum

    @property
    def amplitude_maximum(self):
        """Get the value of amplitude_maximum."""
        return self.mapped["amplitude_maximum"]

    @amplitude_maximum.setter
    def amplitude_maximum(self, amplitude_maximum):
        """Set the value of amplitude_maximum."""
        self.mapped["amplitude_maximum"] = amplitude_maximum

    @property
    def amplitude_bins(self):
        """Get the value of amplitude_bins."""
        return self.mapped["amplitude_bins"]

    @amplitude_bins.setter
    def amplitude_bins(self, amplitude_bins):
        """Set the value of amplitude_bins."""
        self.mapped["amplitude_bins"] = amplitude_bins

    @property
    def amplitude_axis(self):
        """Get the value of amplitude_axis."""
        return self.mapped["amplitude_axis"]

    @amplitude_axis.setter
    def amplitude_axis(self, amplitude_axis):
        """Set the value of amplitude_axis."""
        self.mapped["amplitude_axis"] = amplitude_axis

    @property
    def temperature_minimum(self):
        """Get the value of temperature_minimum."""
        return self.mapped["temperature_minimum"]

    @temperature_minimum.setter
    def temperature_minimum(self, temperature_minimum):
        """Set the value of temperature_minimum."""
        self.mapped["temperature_minimum"] = temperature_minimum

    @property
    def temperature_maximum(self):
        """Get the value of temperature_maximum."""
        return self.mapped["temperature_maximum"]

    @temperature_maximum.setter
    def temperature_maximum(self, temperature_maximum):
        """Set the value of temperature_maximum."""
        self.mapped["temperature_maximum"] = temperature_maximum

    @property
    def temperature_bins(self):
        """Get the value of temperature_bins."""
        return self.mapped["temperature_bins"]

    @temperature_bins.setter
    def temperature_bins(self, temperature_bins):
        """Set the value of temperature_bins."""
        self.mapped["temperature_bins"] = temperature_bins

    @property
    def temperature_axis(self):
        """Get the value of temperature_axis."""
        return self.mapped["temperature_axis"]

    @temperature_axis.setter
    def temperature_axis(self, temperature_axis):
        """Set the value of temperature_axis."""
        self.mapped["temperature_axis"] = temperature_axis

    @property
    def upsample_current(self):
        """Get the value of upsample_current."""
        return self.mapped["upsample_current"]

    @upsample_current.setter
    def upsample_current(self, upsample_current: int):
        """Set the value of upsample_current."""
        self.mapped["upsample_current"] = upsample_current

    @property
    def upsample_voltage(self):
        """Get the value of upsample_voltage."""
        return self.mapped["upsample_voltage"]

    @upsample_voltage.setter
    def upsample_voltage(self, upsample_voltage: int):
        """Set the value of upsample_voltage."""
        self.mapped["upsample_voltage"] = upsample_voltage

    @property
    def upsample_amplitude(self):
        """Get the value of upsample_amplitude."""
        return self.mapped["upsample_amplitude"]

    @upsample_amplitude.setter
    def upsample_amplitude(self, upsample_amplitude: int):
        """Set the value of upsample_amplitude."""
        self.mapped["upsample_amplitude"] = upsample_amplitude

    @property
    def upsample_temperature(self):
        """Get the value of upsample_temperature."""
        return self.mapped["upsample_temperature"]

    @upsample_temperature.setter
    def upsample_temperature(self, upsample_temperature: int):
        """Set the value of upsample_temperature."""
        self.mapped["upsample_temperature"] = upsample_temperature

    # endregion
