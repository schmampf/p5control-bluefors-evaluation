"""
Module iv evaluation, that evaluates data according to p5control-bluefors.
"""

import sys
import logging
import importlib

import numpy as np

from tqdm import tqdm
from scipy import constants

from utilities.baseevaluation import BaseEvaluation
from utilities.basefunctions import linear_fit
from utilities.basefunctions import bin_y_over_x

logger = logging.getLogger(__name__)
importlib.reload(sys.modules["utilities.baseevaluation"])
importlib.reload(sys.modules["utilities.basefunctions"])


class IVEvaluation(BaseEvaluation):
    """
    Description
    """

    def __init__(
        self,
        name="iv eva",
    ):
        """
        Description
        """
        super().__init__()
        self._iv_eva_name = name

        self.eva_current = True
        self.eva_voltage = True
        self.eva_temperature = True

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
            "upsampling_current": None,
            "upsampling_voltage": None,
        }

        self.up_sweep = self.get_empty_dictionary()
        self.down_sweep = self.get_empty_dictionary()

        # self.up_sweep_over_amplitude = self.get_empty_dictionary()
        # self.down_sweep_over_amplitude = self.get_empty_dictionary()

        self.setV(2.0e-3, voltage_bins=100)
        self.setI(1.0e-6, current_bins=100)

        logger.info("(%s) ... IVEvaluation initialized.", self._iv_eva_name)

    def get_empty_dictionary(self, len_y: int = 0):
        logger.debug("(%s) get_empty_dictionary(len_y=%s)", self._iv_eva_name, len_y)
        """
        Returns dictionary with initialized arrays.
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
        voltage_absolute=None,
        voltage_minimum=None,
        voltage_maximum=None,
        voltage_bins=None,
    ):
        """
        Sets voltage-axis. (Optional)

        Parameters
        ----------
        voltage_absolute : float
            voltage_minimum = -voltage_absolute
            voltage_maximum = +voltage_absolute
        voltage_minimum : float
            voltage_minimum, is minimum value on voltage-axis
        voltage_maximum : float
            voltage_maximum, is maximum value on voltage-axis
        voltage_bins : float
            Number of bins minus 1. (float, since default must be np.nan)

        """
        if voltage_absolute is not None:
            voltage_minimum = -voltage_absolute
            voltage_maximum = +voltage_absolute

        if voltage_minimum is not None:
            self.voltage_minimum = voltage_minimum
        if voltage_maximum is not None:
            self.voltage_maximum = voltage_maximum
        if voltage_bins is not None:
            self.voltage_bins = voltage_bins

        # Calculate new V-Axis
        self.voltage_axis = np.linspace(
            self.voltage_minimum,
            self.voltage_maximum,
            int(self.voltage_bins) + 1,
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
        current_absolute=None,
        current_minimum=None,
        current_maximum=None,
        current_bins=None,
    ):
        """
        Sets current-axis. (similar to setV)
        """
        if current_absolute is not None:
            current_minimum = -current_absolute
            current_maximum = +current_absolute

        if current_minimum is not None:
            self.current_minimum = current_minimum
        if current_maximum is not None:
            self.current_maximum = current_maximum
        if current_bins is not None:
            self.current_bins = current_bins

        self.current_axis = np.linspace(
            self.current_minimum,
            self.current_maximum,
            int(self.current_bins) + 1,
        )

        logger.debug(
            "(%s) setI(%s, %s, %s)",
            self._iv_eva_name,
            self.current_minimum,
            self.current_maximum,
            self.current_bins,
        )

    def get_current_voltage(self, data_file, specific_key):
        logger.debug(
            "(%s) get_current_voltage(%s, %s)",
            self._iv_eva_name,
            data_file,
            specific_key,
        )

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

    def get_temperatures(self, data_file, key):
        logger.debug(
            "(%s) get_temperatures(, %s, %s)", self._iv_eva_name, data_file, key
        )
        # Retrieve Temperature Dataset
        T_data_set = data_file.get(f"measurement/{self.measurement_key}/{key}/sweep")
        check = False
        if "bluefors" in T_data_set.keys():  # type: ignore
            measurement_data_temperature = np.array(
                data_file.get(
                    f"measurement/{self.measurement_key}/{key}/sweep/bluefors"
                )
            )
            temporary_time = measurement_data_temperature["time"]
            temporary_temperature = measurement_data_temperature["Tsample"]
            check = True
        else:
            status = np.array(data_file.get("status/bluefors/temperature/MCBJ"))
            temporary_time = status["time"]
            temporary_temperature = status["T"]

        return temporary_time, temporary_temperature, check

    def get_iv_binning_done(
        self,
        index,
        specific_trigger,
        dictionary,
        v_raw,
        i_raw,
        trigger,
        time,
    ):
        logger.debug("(%s) get_iv_binning_done(...)", self._iv_eva_name)
        # Get time
        time = time[trigger == specific_trigger]
        dictionary["time_start"][index] = time[0]
        dictionary["time_stop"][index] = time[-1]

        # Get V, I
        v_raw = v_raw[trigger == specific_trigger]
        i_raw = i_raw[trigger == specific_trigger]
        dictionary["iv_tuples"][index] = [i_raw, v_raw]

        if self.eva_current:
            # Bin that stuff
            dictionary["current"][index, :], _ = bin_y_over_x(
                v_raw,
                i_raw,
                self.voltage_axis,
                upsampling=self.upsampling_current,
            )
            dictionary["time_current"][index, :], _ = bin_y_over_x(
                v_raw,
                time,
                self.voltage_axis,
                upsampling=self.upsampling_current,
            )

        if self.eva_voltage:
            dictionary["voltage"][index, :], _ = bin_y_over_x(
                i_raw,
                v_raw,
                self.current_axis,
                upsampling=self.upsampling_voltage,
            )
            dictionary["time_voltage"][index, :], _ = bin_y_over_x(
                i_raw,
                time,
                self.current_axis,
                upsampling=self.upsampling_voltage,
            )

    def get_temperature_binning_done(
        self,
        index,
        dictionary,
        temporary_time,
        temporary_temperature,
    ):
        logger.debug("(%s) get_temperature_binning_done(...)", self._iv_eva_name)
        if self.eva_current:
            temporary_time_current = linear_fit(dictionary["time_current"][index])

            if temporary_time_current[0] > temporary_time_current[1]:
                temporary_time_current = np.flip(temporary_time_current)

            dictionary["temperature_current"][index, :], _ = bin_y_over_x(
                temporary_time,
                temporary_temperature,
                temporary_time_current,
                upsampling=1000,
            )
        if self.eva_voltage:
            temporary_time_voltage = linear_fit(dictionary["time_voltage"][index])

            if temporary_time_voltage[0] > temporary_time_voltage[1]:
                temporary_time_voltage = np.flip(temporary_time_voltage)

            dictionary["temperature_voltage"][index, :], _ = bin_y_over_x(
                temporary_time,
                temporary_temperature,
                temporary_time_voltage,
                upsampling=1000,
            )

    def sort_mapped_over_y(self, y_bounds):
        logger.debug("(%s) sort_mapped_over_y(%s)", self._iv_eva_name, y_bounds)
        indices = np.argsort(self.y_unsorted)
        self.sorting_indices = indices

        self.y_axis = self.y_unsorted[indices]
        self.voltage_offset_1 = self.voltage_offset_1[indices]
        self.voltage_offset_2 = self.voltage_offset_2[indices]

        if y_bounds is not None:
            self.y_axis = self.y_axis[y_bounds[0] : y_bounds[1]]
            self.voltage_offset_1 = self.voltage_offset_1[y_bounds[0] : y_bounds[1]]
            self.voltage_offset_2 = self.voltage_offset_2[y_bounds[0] : y_bounds[1]]
        return indices

    def sort_dictionary_over_y(self, indices, y_bounds, dictionary):
        logger.debug(
            "(%s) sort_dictionary_over_y(%s, %s, %s)",
            self._iv_eva_name,
            indices,
            y_bounds,
            dictionary,
        )
        dictionary["time_start"] = dictionary["time_start"][indices]
        dictionary["time_stop"] = dictionary["time_stop"][indices]
        if y_bounds is not None:
            dictionary["time_start"] = dictionary["time_start"][
                y_bounds[0] : y_bounds[1]
            ]
            dictionary["time_stop"] = dictionary["time_stop"][y_bounds[0] : y_bounds[1]]

        if self.eva_current:
            dictionary["current"] = dictionary["current"][indices, :]
            dictionary["time_current"] = dictionary["time_current"][indices, :]
            if self.eva_temperature:
                dictionary["temperature_current"] = dictionary["temperature_current"][
                    indices, :
                ]
            if y_bounds is not None:
                dictionary["current"] = dictionary["current"][
                    y_bounds[0] : y_bounds[1], :
                ]
                dictionary["time_current"] = dictionary["time_current"][
                    y_bounds[0] : y_bounds[1], :
                ]
                if self.eva_temperature:
                    dictionary["temperature_current"] = dictionary[
                        "temperature_current"
                    ][y_bounds[0] : y_bounds[1], :]

        if self.eva_voltage:
            dictionary["voltage"] = dictionary["voltage"][indices, :]
            dictionary["time_voltage"] = dictionary["time_voltage"][indices, :]
            if self.eva_temperature:
                dictionary["temperature_voltage"] = dictionary["temperature_voltage"][
                    indices, :
                ]
            if y_bounds is not None:
                dictionary["voltage"] = dictionary["voltage"][
                    y_bounds[0] : y_bounds[1], :
                ]
                dictionary["time_voltage"] = dictionary["time_voltage"][
                    y_bounds[0] : y_bounds[1], :
                ]
                if self.eva_temperature:
                    dictionary["temperature_voltage"] = dictionary[
                        "temperature_voltage"
                    ][y_bounds[0] : y_bounds[1], :]
        return

    def get_differentials(self, dictionary):
        logger.debug("(%s) get_differentials(%s)", self._iv_eva_name, dictionary)
        conductance_quantum = constants.physical_constants["conductance quantum"][0]
        if self.eva_current:
            dictionary["differential_conductance"] = (
                np.gradient(dictionary["current"], self.voltage_axis, axis=1)
                / conductance_quantum
            )
            dictionary["temperature"] = np.nanmean(
                dictionary["temperature_current"], axis=1
            )
        if self.eva_voltage:
            dictionary["differential_resistance"] = np.gradient(
                dictionary["voltage"], self.current_axis, axis=1
            )
            dictionary["temperature"] = np.nanmean(
                dictionary["temperature_voltage"], axis=1
            )

    def getSingleIV(self, y_key: str, specific_trigger: int = 1):
        logger.debug(
            "(%s) getSingleIV(%s, %s)", self._iv_eva_name, y_key, specific_trigger
        )
        single_iv = self.get_empty_dictionary(len_y=1)
        data_file = self.accessFile()
        (
            v_raw,
            i_raw,
            trigger,
            time,
            single_iv["voltage_offset_1"],
            single_iv["voltage_offset_2"],
        ) = self.get_current_voltage(data_file, y_key)
        self.get_iv_binning_done(
            0,
            specific_trigger,
            single_iv,
            v_raw,
            i_raw,
            trigger,
            time,
        )
        if self.eva_temperature:
            temporary_time, temporary_temperature, _ = self.get_temperatures(
                data_file, y_key
            )
            self.get_temperature_binning_done(
                0,
                single_iv,
                temporary_time,
                temporary_temperature,
            )
        self.get_differentials(single_iv)
        return single_iv

    def getMaps(
        self,
        y_bounds=None,
    ):
        """getMaps()
        - Calculate I and V and split in up / down sweep
        - Maps I, differential_conductance, t over linear V-axis
        - Also saves start and stop times
        - As well as offsets
        - sort by y-axis
        """

        logger.info("(%s) getMaps()", self._iv_eva_name)

        # Initialize voltage_offset_values
        len_y = np.shape(self.y_unsorted)[0]
        self.voltage_offset_1 = np.full(len_y, np.nan, dtype="float64")
        self.voltage_offset_2 = np.full(len_y, np.nan, dtype="float64")

        if self.index_trigger_up is not None:
            self.up_sweep = self.get_empty_dictionary()
        if self.index_trigger_down is not None:
            self.down_sweep = self.get_empty_dictionary()

        # Access File
        data_file = self.accessFile()

        # check if to_pop are there and get Single IVs
        if self.y_0_key != "":
            if self.index_trigger_up is not None:
                self.up_sweep["plain"] = self.getSingleIV(
                    self.y_0_key, self.index_trigger_up
                )
            if self.index_trigger_down is not None:
                self.down_sweep["plain"] = self.getSingleIV(
                    self.y_0_key, self.index_trigger_down
                )

        # check if temperature data is availabel
        temporary_time, temporary_temperature, check = self.get_temperatures(
            data_file, self.specific_keys[0]
        )

        # Iterate over Keys
        for index, key in enumerate(tqdm(self.specific_keys)):
            key = self.specific_keys[index]

            # Calculate I, V from ADwin data
            (
                v_raw,
                i_raw,
                trigger,
                time,
                self.voltage_offset_1[index],
                self.voltage_offset_2[index],
            ) = self.get_current_voltage(data_file, key)

            # Get binning done
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

            if self.eva_temperature:
                if check:
                    temporary_time, temporary_temperature, check = (
                        self.get_temperatures(data_file, key)
                    )

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

        indices = self.sort_mapped_over_y(y_bounds)

        if self.index_trigger_up is not None:
            self.sort_dictionary_over_y(indices, y_bounds, self.up_sweep)

        if self.index_trigger_down is not None:
            self.sort_dictionary_over_y(indices, y_bounds, self.down_sweep)

        # calculating differential conductance & resistance
        if self.index_trigger_up is not None:
            self.get_differentials(self.up_sweep)
        if self.index_trigger_down is not None:
            self.get_differentials(self.down_sweep)

    @property
    def eva_current(self):
        """get eva_current"""
        return self.base_evaluation["eva_current"]

    @eva_current.setter
    def eva_current(self, eva_current):
        """set eva_current"""
        self.base_evaluation["eva_current"] = eva_current
        logger.debug("(%s) eva_current = %s", self._iv_eva_name, eva_current)

    @property
    def eva_voltage(self):
        """get eva_voltage"""
        return self.base_evaluation["eva_voltage"]

    @eva_voltage.setter
    def eva_voltage(self, eva_voltage):
        """set eva_voltage"""
        self.base_evaluation["eva_voltage"] = eva_voltage
        logger.debug("(%s) eva_voltage = %s", self._iv_eva_name, eva_voltage)

    @property
    def eva_temperature(self):
        """get eva_temperature"""
        return self.base_evaluation["eva_temperature"]

    @eva_temperature.setter
    def eva_temperature(self, eva_temperature):
        """set eva_temperature"""
        self.base_evaluation["eva_temperature"] = eva_temperature
        logger.debug("(%s) eva_temperature = %s", self._iv_eva_name, eva_temperature)

    ### IV Properties ###

    @property
    def voltage_offset_1(self):
        return self.mapped["voltage_offset_1"]

    @voltage_offset_1.setter
    def voltage_offset_1(self, voltage_offset_1):
        self.mapped["voltage_offset_1"] = voltage_offset_1

    @property
    def voltage_offset_2(self):
        return self.mapped["voltage_offset_2"]

    @voltage_offset_2.setter
    def voltage_offset_2(self, voltage_offset_2):
        self.mapped["voltage_offset_2"] = voltage_offset_2

    @property
    def y_axis(self):
        return self.mapped["y_axis"]

    @y_axis.setter
    def y_axis(self, y_axis):
        self.mapped["y_axis"] = y_axis

    @property
    def voltage_minimum(self):
        return self.mapped["voltage_minimum"]

    @voltage_minimum.setter
    def voltage_minimum(self, voltage_minimum):
        self.mapped["voltage_minimum"] = voltage_minimum

    @property
    def voltage_maximum(self):
        return self.mapped["voltage_maximum"]

    @voltage_maximum.setter
    def voltage_maximum(self, voltage_maximum):
        self.mapped["voltage_maximum"] = voltage_maximum

    @property
    def voltage_bins(self):
        return self.mapped["voltage_bins"]

    @voltage_bins.setter
    def voltage_bins(self, voltage_bins):
        self.mapped["voltage_bins"] = voltage_bins

    @property
    def voltage_axis(self):
        return self.mapped["voltage_axis"]

    @voltage_axis.setter
    def voltage_axis(self, voltage_axis):
        self.mapped["voltage_axis"] = voltage_axis

    @property
    def current_minimum(self):
        return self.mapped["current_minimum"]

    @current_minimum.setter
    def current_minimum(self, current_minimum):
        self.mapped["current_minimum"] = current_minimum

    @property
    def current_maximum(self):
        return self.mapped["current_maximum"]

    @current_maximum.setter
    def current_maximum(self, current_maximum):
        self.mapped["current_maximum"] = current_maximum

    @property
    def current_bins(self):
        return self.mapped["current_bins"]

    @current_bins.setter
    def current_bins(self, current_bins):
        self.mapped["current_bins"] = current_bins

    @property
    def current_axis(self):
        return self.mapped["current_axis"]

    @current_axis.setter
    def current_axis(self, current_axis):
        self.mapped["current_axis"] = current_axis

    @property
    def upsampling_current(self):
        return self.mapped["upsampling_current"]

    @upsampling_current.setter
    def upsampling_current(self, upsampling_current):
        self.mapped["upsampling_current"] = upsampling_current

    @property
    def upsampling_voltage(self):
        return self.mapped["upsampling_voltage"]

    @upsampling_voltage.setter
    def upsampling_voltage(self, upsampling_voltage):
        self.mapped["upsampling_voltage"] = upsampling_voltage
