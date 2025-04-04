"""
Module a base evaluation, that evaluates data according to p5control-bluefors.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from h5py import File
from scipy import constants

from utilities.baseclass import BaseClass
from utilities.basefunctions import linear_fit
from utilities.basefunctions import bin_y_over_x
from utilities.basefunctions import bin_z_over_y
from utilities.key_database import POSSIBLE_MEASUREMENT_KEYS

plt.ioff()

logger = logging.getLogger(__name__)


class BaseEvaluation(BaseClass):
    """
    Description
    """

    def __init__(
        self,
        name="base eva",
    ):
        """
        Description
        """
        super().__init__(name=name)

        # initialize key lists
        self.possible_measurement_keys = POSSIBLE_MEASUREMENT_KEYS

        # initialize general stuff
        self.general = {
            "voltage_amplification_1": 1,
            "voltage_amplification_2": 1,
            "reference_resistor": 51.689e3,
            "index_trigger_up": 1,
            "index_trigger_down": 2,
            "measurement_key": "",
            "specific_keys": [],
            "y_unsorted": np.array([]),
            "upsampling": None,
        }

        # initilize voltage axis
        voltage_minimum = -1.8e-3
        voltage_maximum = +1.8e-3
        voltage_bins = 900
        voltage_axis = np.linspace(
            voltage_minimum,
            voltage_maximum,
            int(voltage_bins) + 1,
        )
        self.mapped = {
            "voltage_minimum": voltage_minimum,
            "voltage_maximum": voltage_maximum,
            "voltage_bins": voltage_bins,
            "voltage_axis": voltage_axis,
            "y_axis": np.array([]),
            "voltage_offset_1": np.array([]),
            "voltage_offset_2": np.array([]),
            "current_up": np.array([]),
            "current_down": np.array([]),
            "differential_conductance_up": np.array([]),
            "differential_conductance_down": np.array([]),
            "temperature_all_up": np.array([]),
            "temperature_all_down": np.array([]),
            "temperature_mean_up": np.array([]),
            "temperature_mean_down": np.array([]),
            "time_up": np.array([]),
            "time_down": np.array([]),
            "time_up_start": np.array([]),
            "time_up_stop": np.array([]),
            "time_down_start": np.array([]),
            "time_down_stop": np.array([]),
        }

        # initialize temperature axis
        temperature_minimum = 0
        temperature_maximum = 2
        temperature_bins = 2000
        temperature_axis = np.linspace(
            temperature_minimum,
            temperature_maximum,
            int(temperature_bins) + 1,
        )
        self.mapped_over_temperature = {
            "temperature_minimum": temperature_minimum,
            "temperature_maximum": temperature_maximum,
            "temperature_bins": temperature_bins,
            "temperature_axis": temperature_axis,
            "counter_up": np.array([]),
            "counter_down": np.array([]),
            "current_up": np.array([]),
            "current_down": np.array([]),
            "differential_conductance_up": np.array([]),
            "differential_conductance_down": np.array([]),
            "y_axis_up": np.array([]),
            "y_axis_down": np.array([]),
        }

        logger.info("(%s) ... BaseEvaluation initialized.", self._name)

    def showAmplifications(
        self,
    ):
        """
        Shows amplification over time during whole measurement.
        """
        logger.info("(%s) showAmplifications()", self._name)

        file_name = (
            self.base["file_directory"]
            + self.base["file_folder"]
            + self.base["file_name"]
        )
        data_file = File(file_name, "r")
        femto_data = np.array(data_file.get("status/femto"))
        time = femto_data["time"]
        amplification_a = femto_data["amp_A"]
        amplification_b = femto_data["amp_B"]
        plt.close("all")
        plt.figure(1000, figsize=(6, 1.5))
        plt.semilogy(time, amplification_a, "-", label="voltage amplification 1")
        plt.semilogy(time, amplification_b, "--", label="voltage amplification 2")
        plt.legend()
        plt.title("Femto Amplifications according to Status")
        plt.xlabel("time (s)")
        plt.ylabel("amplification")
        plt.tight_layout()
        plt.show()

    def setAmplifications(
        self,
        voltage_amplification_1: float,
        voltage_amplification_2: float,
    ):
        """
        Sets Amplifications for Calculations.
        """
        logger.info(
            "(%s) setAmplifications(%s, %s)",
            self._name,
            voltage_amplification_1,
            voltage_amplification_2,
        )
        self.general["voltage_amplification_1"] = voltage_amplification_1
        self.general["voltage_amplification_2"] = voltage_amplification_2

    def showMeasurements(self):
        """
        Shows available Measurements in File.
        """
        logger.info("(%s) showMeasurements()", self._name)

        file_name = (
            self.base["file_directory"]
            + self.base["file_folder"]
            + self.base["file_name"]
        )
        data_file = File(file_name, "r")
        liste = list(data_file["measurement"].keys())  # type: ignore
        logger.info("(%s) %s", self._name, liste)

    def setMeasurement(self, measurement_key: str):
        """
        Sets Measurement Key. Mandatory for further Evaluation.

        Parameters
        ----------
        measurement_key : str
            name of measurement
        """
        logger.info("(%s) setMeasurement('%s')", self._name, measurement_key)
        try:

            file_name = (
                self.base["file_directory"]
                + self.base["file_folder"]
                + self.base["file_name"]
            )
            data_file = File(file_name, "r")
            measurement_data = data_file.get(f"measurement/{measurement_key}")
            self.general["specific_keys"] = list(measurement_data)  # type: ignore
            self.general["measurement_key"] = measurement_key
        except KeyError:
            self.general["specific_keys"] = []
            self.general["measurement_key"] = ""
            logger.error("(%s) '%s' found in File.", self._name, measurement_key)

    def showKeys(self):
        """
        Shows available Keys in Measurement.
        """
        logger.info("(%s) showKeys()", self._name)
        show_keys = (
            self.general["specific_keys"][:2] + self.general["specific_keys"][-2:]
        )
        logger.info("(%s) %s", self._name, show_keys)

    def setKeys(
        self,
        parameters=None,
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

        """
        if self.general["measurement_key"] == "":
            logger.warning("(%s) Do setMeasurement() first.", self._name)
            return

        if parameters is None:
            parameters = self.possible_measurement_keys[self.general["measurement_key"]]

        logger.info("(%s) setKeys(%s)", self._name, parameters)

        try:
            i0 = parameters[0]
            i1 = parameters[1]
            norm = parameters[2]
            to_pop = parameters[3]
        except IndexError:
            logger.warning("(%s) List of Parameter is incompete.", self._name)
            return

        if to_pop in self.general["specific_keys"]:
            self.general["specific_keys"].remove(to_pop)
        else:
            logger.warning("(%s) Key to pop is not found.", self._name)

        y = []
        for key in self.general["specific_keys"]:
            temp = key[i0:i1]
            temp = float(temp) * norm
            y.append(temp)
        y = np.array(y)

        self.general["y_unsorted"] = y

    def addKey(
        self,
        key: str,
        y_value: float,
    ):
        """
        Adds Key to Specific Key List and Y-Value to unsorted Y-Values.

        Parameters
        ----------
        key : str
            key to add to specific keys
        y_value : float
            y_value to add to y_unsorted array
        """
        logger.info("(%s) addKey(%s, %s)", self._name, key, y_value)

        self.general["specific_keys"].append(key)
        self.general["y_unsorted"] = np.concatenate(
            (self.general["y_unsorted"], [y_value])
        )

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
            self.mapped["voltage_minimum"] = voltage_minimum
        if voltage_maximum is not None:
            self.mapped["voltage_maximum"] = voltage_maximum
        if voltage_bins is not None:
            self.mapped["voltage_bins"] = voltage_bins

        # Calculate new V-Axis
        self.mapped["voltage_axis"] = np.linspace(
            self.mapped["voltage_minimum"],
            self.mapped["voltage_maximum"],
            int(self.mapped["voltage_bins"]) + 1,
        )

        logger.info(
            "(%s) setV(%s, %s, %s)",
            self._name,
            self.mapped["voltage_minimum"],
            self.mapped["voltage_maximum"],
            self.mapped["voltage_bins"],
        )

    def getMaps(
        self,
        bounds=None,
    ):
        """getMaps()
        - Calculate I and V and split in up / down sweep
        - Maps I, differential_conductance, T over linear V-axis
        - Also saves start and stop times
        - As well as offsets
        - sort by y-axis
        """

        logger.info("(%s) getMaps()", self._name)

        # Access File
        try:
            file_name = (
                self.base["file_directory"]
                + self.base["file_folder"]
                + self.base["file_name"]
            )
            data_file = File(file_name, "r")
        except AttributeError:
            logger.error("(%s) File can not be found!", self._name)
            return
        except KeyError:
            logger.error("(%s) Measurement can not be found!", self._name)
            return

        len_voltage = np.shape(self.mapped["voltage_axis"])[0]
        len_y = np.shape(self.general["y_unsorted"])[0]

        # Initialize all values
        self.mapped["voltage_offset_1"] = np.full(len_y, np.nan, dtype="float64")
        self.mapped["voltage_offset_2"] = np.full(len_y, np.nan, dtype="float64")

        if self.general["index_trigger_up"] is not None:
            self.mapped["current_up"] = np.full(
                (len_y, len_voltage), np.nan, dtype="float64"
            )
            self.mapped["time_up"] = np.full(
                (len_y, len_voltage), np.nan, dtype="float64"
            )
            self.mapped["time_up_start"] = np.full(len_y, np.nan, dtype="float64")
            self.mapped["time_up_stop"] = np.full(len_y, np.nan, dtype="float64")

            self.mapped["temperature_all_up"] = np.full(
                (len_y, len_voltage), np.nan, dtype="float64"
            )

        if self.general["index_trigger_down"] is not None:
            self.mapped["current_down"] = np.full(
                (len_y, len_voltage), np.nan, dtype="float64"
            )
            self.mapped["time_down"] = np.full(
                (len_y, len_voltage), np.nan, dtype="float64"
            )
            self.mapped["temperature_all_down"] = np.full(
                (len_y, len_voltage), np.nan, dtype="float64"
            )
            self.mapped["time_down_start"] = np.full(len_y, np.nan, dtype="float64")
            self.mapped["time_down_stop"] = np.full(len_y, np.nan, dtype="float64")

        # Iterate over Keys
        for i, k in enumerate(tqdm(self.general["specific_keys"])):

            # Retrieve Offset Dataset
            measurement_data_offset = np.array(
                data_file.get(
                    f"measurement/{self.general['measurement_key']}/{k}/offset/adwin"
                )
            )
            # Calculate Offsets
            self.mapped["voltage_offset_1"][i] = np.nanmean(
                np.array(measurement_data_offset["V1"])
            )
            self.mapped["voltage_offset_2"][i] = np.nanmean(
                np.array(measurement_data_offset["V2"])
            )

            # Retrieve Sweep Dataset
            measurement_data_sweep = np.array(
                data_file.get(
                    f"measurement/{self.general['measurement_key']}/{k}/sweep/adwin"
                )
            )
            # Get Voltage Readings of Adwin
            trigger = np.array(measurement_data_sweep["trigger"], dtype="int")
            time = np.array(measurement_data_sweep["time"], dtype="float64")
            v1 = np.array(measurement_data_sweep["V1"], dtype="float64")
            v2 = np.array(measurement_data_sweep["V2"], dtype="float64")

            # Calculate V, I
            v_raw = (v1 - self.mapped["voltage_offset_1"][i]) / self.general[
                "voltage_amplification_1"
            ]
            i_raw = (
                (v2 - self.mapped["voltage_offset_2"][i])
                / self.general["voltage_amplification_2"]
                / self.general["reference_resistor"]
            )

            if self.general["index_trigger_up"] is not None:
                # Get upsweep
                v_raw_up = v_raw[trigger == self.general["index_trigger_up"]]
                i_raw_up = i_raw[trigger == self.general["index_trigger_up"]]
                time_up = time[trigger == self.general["index_trigger_up"]]
                # Calculate Timepoints
                self.mapped["time_up_start"][i] = time_up[0]
                self.mapped["time_up_stop"][i] = time_up[-1]
                # Bin that stuff
                i_up, _ = bin_y_over_x(
                    v_raw_up,
                    i_raw_up,
                    self.mapped["voltage_axis"],
                    upsampling=self.upsampling,
                )
                time_up, _ = bin_y_over_x(
                    v_raw_up,
                    time_up,
                    self.mapped["voltage_axis"],
                    upsampling=self.upsampling,
                )
                # Save to Array
                self.mapped["current_up"][i, :] = i_up
                self.mapped["time_up"][i, :] = time_up

            if self.general["index_trigger_down"] is not None:
                # Get dwonsweep
                v_raw_down = v_raw[trigger == self.general["index_trigger_down"]]
                i_raw_down = i_raw[trigger == self.general["index_trigger_down"]]
                time_down = time[trigger == self.general["index_trigger_down"]]
                # Calculate Timepoints
                self.mapped["time_down_start"][i] = time_down[0]
                self.mapped["time_down_stop"][i] = time_down[-1]
                # Bin that stuff
                i_down, _ = bin_y_over_x(
                    v_raw_down,
                    i_raw_down,
                    self.mapped["voltage_axis"],
                    upsampling=self.upsampling,
                )
                time_down, _ = bin_y_over_x(
                    v_raw_down,
                    time_down,
                    self.mapped["voltage_axis"],
                    upsampling=self.upsampling,
                )
                # Save to Array
                self.mapped["current_down"][i, :] = i_down
                self.mapped["time_down"][i, :] = time_down

            # Retrieve Temperature Dataset
            data_set = data_file.get(
                f"measurement/{self.general['measurement_key']}/{k}/sweep"
            )
            if "bluefors" in data_set.keys():  # type: ignore
                measurement_data_temperature = np.array(
                    data_file.get(
                        f"measurement/{self.general['measurement_key']}/{k}/sweep/bluefors"
                    )
                )
                temporary_time = measurement_data_temperature["time"]
                temporary_temperature = measurement_data_temperature["Tsample"]

                if self.general["index_trigger_up"] is not None:
                    temporary_time_up = linear_fit(time_up)
                    if temporary_time_up[0] > temporary_time_up[1]:
                        temporary_time_up = np.flip(temporary_time_up)
                    temperature_up, _ = bin_y_over_x(
                        temporary_time,
                        temporary_temperature,
                        temporary_time_up,
                        upsampling=1000,
                    )
                    self.mapped["temperature_all_up"][i, :] = temperature_up

                if self.general["index_trigger_down"] is not None:
                    temporary_time_down = linear_fit(time_down)
                    if temporary_time_down[0] > temporary_time_down[1]:
                        temporary_time_down = np.flip(temporary_time_down)
                    temperature_down, _ = bin_y_over_x(
                        temporary_time,
                        temporary_temperature,
                        temporary_time_down,
                        upsampling=1000,
                    )
                    self.mapped["temperature_all_down"][i, :] = temperature_down
            else:
                measurement_data_temperature = False
                logger.error("(%s) No temperature data available!", self._name)

        # sorting afterwards, because of probably unknown characters in keys
        indices = np.argsort(self.general["y_unsorted"])
        self.mapped["sorting_indices"] = indices
        self.mapped["bounds"] = bounds

        self.mapped["y_axis"] = self.general["y_unsorted"][indices]
        self.mapped["voltage_offset_1"] = self.mapped["voltage_offset_1"][indices]
        self.mapped["voltage_offset_2"] = self.mapped["voltage_offset_2"][indices]

        if self.general["index_trigger_up"] is not None:
            self.mapped["current_up"] = self.mapped["current_up"][indices, :]
            self.mapped["time_up"] = self.mapped["time_up"][indices, :]
            self.mapped["temperature_all_up"] = self.mapped["temperature_all_up"][
                indices, :
            ]
            self.mapped["time_up_start"] = self.mapped["time_up_start"][indices]
            self.mapped["time_up_stop"] = self.mapped["time_up_stop"][indices]

        if self.general["index_trigger_down"] is not None:
            self.mapped["current_down"] = self.mapped["current_down"][indices, :]
            self.mapped["time_down"] = self.mapped["time_down"][indices, :]
            self.mapped["temperature_all_down"] = self.mapped["temperature_all_down"][
                indices, :
            ]
            self.mapped["time_down_start"] = self.mapped["time_down_start"][indices]
            self.mapped["time_down_stop"] = self.mapped["time_down_stop"][indices]

        if bounds is not None:
            self.mapped["y_axis"] = self.mapped["y_axis"][bounds[0] : bounds[1]]
            self.mapped["voltage_offset_1"] = self.mapped["voltage_offset_1"][
                bounds[0] : bounds[1]
            ]
            self.mapped["voltage_offset_2"] = self.mapped["voltage_offset_2"][
                bounds[0] : bounds[1]
            ]

            if self.general["index_trigger_up"] is not None:
                self.mapped["current_up"] = self.mapped["current_up"][
                    bounds[0] : bounds[1], :
                ]
                self.mapped["time_up"] = self.mapped["time_up"][
                    bounds[0] : bounds[1], :
                ]
                self.mapped["temperature_all_up"] = self.mapped["temperature_all_up"][
                    bounds[0] : bounds[1], :
                ]
                self.mapped["time_up_start"] = self.mapped["time_up_start"][
                    bounds[0] : bounds[1]
                ]
                self.mapped["time_up_stop"] = self.mapped["time_up_stop"][
                    bounds[0] : bounds[1]
                ]

            if self.general["index_trigger_down"] is not None:
                self.mapped["current_down"] = self.mapped["current_down"][
                    bounds[0] : bounds[1], :
                ]
                self.mapped["time_down"] = self.mapped["time_down"][
                    bounds[0] : bounds[1], :
                ]
                self.mapped["temperature_all_down"] = self.mapped[
                    "temperature_all_down"
                ][bounds[0] : bounds[1], :]
                self.mapped["time_down_start"] = self.mapped["time_down_start"][
                    bounds[0] : bounds[1]
                ]
                self.mapped["time_down_stop"] = self.mapped["time_down_stop"][
                    bounds[0] : bounds[1]
                ]

        conductance_quantum = constants.physical_constants["conductance quantum"][0]

        if self.general["index_trigger_up"] is not None:
            # calculating differential conductance
            self.mapped["differential_conductance_up"] = (
                np.gradient(
                    self.mapped["current_up"], self.mapped["voltage_axis"], axis=1
                )
                / conductance_quantum
            )
            # calculates self.temperature_mean_up, self.temperature_mean_down
            self.mapped["temperature_mean_up"] = np.nanmean(
                self.mapped["temperature_all_up"], axis=1
            )

        if self.general["index_trigger_down"] is not None:
            self.mapped["differential_conductance_down"] = (
                np.gradient(
                    self.mapped["current_down"], self.mapped["voltage_axis"], axis=1
                )
                / conductance_quantum
            )
            self.mapped["temperature_mean_down"] = np.nanmean(
                self.mapped["temperature_all_down"], axis=1
            )

    def setT(
        self,
        temperature_minimum: float = np.nan,
        temperature_maximum: float = np.nan,
        temperature_bins: float = np.nan,
    ):
        """
        Sets T-axis. (Optional)

        Parameters
        ----------
        temperature_minimum : float
            temperature_minimum, is minimum value on V-axis
        temperature_maximum : float
            temperature_maximum, is minimum value on V-axis
        temperature_bins : float
            Number of bins minus 1. (float, since default must be np.nan)

        """
        if not np.isnan(temperature_minimum):
            self.mapped_over_temperature["temperature_minimum"] = temperature_minimum
        if not np.isnan(temperature_maximum):
            self.mapped_over_temperature["temperature_maximum"] = temperature_maximum
        if not np.isnan(temperature_bins):
            self.mapped_over_temperature["temperature_bins"] = temperature_bins

        # Calculate new Temperature-Axis
        self.mapped_over_temperature["temperature_axis"] = np.linspace(
            self.mapped_over_temperature["temperature_minimum"],
            self.mapped_over_temperature["temperature_maximum"],
            int(self.mapped_over_temperature["temperature_bins"]) + 1,
        )

        logger.info(
            "(%s) setT(%s, %s, %s)",
            self._name,
            self.mapped_over_temperature["temperature_minimum"],
            self.mapped_over_temperature["temperature_maximum"],
            self.mapped_over_temperature["temperature_bins"],
        )

    def getMapsT(
        self,
    ):
        """getMapsT()
        - Maps I, differential_conductance over linear T-axis
        """
        logger.info("(%s) getMapsT()", self._name)

        if self.general["index_trigger_up"] is not None:
            (
                self.mapped_over_temperature["current_up"],
                self.mapped_over_temperature["counter_up"],
            ) = bin_z_over_y(
                self.mapped["temperature_mean_up"],
                self.mapped["current_up"],
                self.mapped_over_temperature["temperature_axis"],
            )
            (
                self.mapped_over_temperature["differential_conductance_up"],
                _,
            ) = bin_z_over_y(
                self.mapped["temperature_mean_up"],
                self.mapped["differential_conductance_up"],
                self.mapped_over_temperature["temperature_axis"],
            )
            self.mapped_over_temperature["y_axis_up"], _ = bin_y_over_x(
                self.mapped["temperature_mean_up"],
                self.mapped["y_axis"],
                self.mapped_over_temperature["temperature_axis"],
            )

        if self.general["index_trigger_down"] is not None:
            (
                self.mapped_over_temperature["current_down"],
                self.mapped_over_temperature["counter_down"],
            ) = bin_z_over_y(
                self.mapped["temperature_mean_down"],
                self.mapped["current_down"],
                self.mapped_over_temperature["temperature_axis"],
            )
            (
                self.mapped_over_temperature["differential_conductance_down"],
                _,
            ) = bin_z_over_y(
                self.mapped["temperature_mean_down"],
                self.mapped["differential_conductance_down"],
                self.mapped_over_temperature["temperature_axis"],
            )
            self.mapped_over_temperature["y_axis_down"], _ = bin_y_over_x(
                self.mapped["temperature_mean_down"],
                self.mapped["y_axis"],
                self.mapped_over_temperature["temperature_axis"],
            )

    @property
    def voltage_amplification_1(self):
        """get voltage_amplification_1"""
        return self.general["voltage_amplification_1"]

    @voltage_amplification_1.setter
    def voltage_amplification_1(self, voltage_amplification_1: int):
        """set voltage_amplification_1"""
        self.general["voltage_amplification_1"] = voltage_amplification_1
        logger.info(
            "(%s) voltage_amplification_1 = %s", self._name, voltage_amplification_1
        )

    @property
    def voltage_amplification_2(self):
        """get voltage_amplification_2"""
        return self.general["voltage_amplification_2"]

    @voltage_amplification_2.setter
    def voltage_amplification_2(self, voltage_amplification_2: int):
        """set voltage_amplification_2"""
        self.general["voltage_amplification_2"] = voltage_amplification_2
        logger.info(
            "(%s) voltage_amplification_2 = %s", self._name, voltage_amplification_2
        )

    @property
    def reference_resistor(self):
        """get reference_resistor"""
        return self.general["reference_resistor"]

    @reference_resistor.setter
    def reference_resistor(self, reference_resistor: float):
        """set reference_resistor"""
        self.general["reference_resistor"] = reference_resistor
        logger.info("(%s) reference_resistor = %s", self._name, reference_resistor)

    @property
    def index_trigger_up(self):
        """get index_trigger_up"""
        return self.general["index_trigger_up"]

    @index_trigger_up.setter
    def index_trigger_up(self, index_trigger_up: int):
        """set index_trigger_up"""
        self.general["index_trigger_up"] = index_trigger_up
        logger.info("(%s) index_trigger_up = %s", self._name, index_trigger_up)

    @property
    def index_trigger_down(self):
        """get index_trigger_down"""
        return self.general["index_trigger_down"]

    @index_trigger_down.setter
    def index_trigger_down(self, index_trigger_down: int):
        """set index_trigger_down"""
        self.general["index_trigger_down"] = index_trigger_down
        logger.info("(%s) index_trigger_down = %s", self._name, index_trigger_down)

    @property
    def measurement_key(self):
        """get measurement_key"""
        return self.general["measurement_key"]

    @measurement_key.setter
    def measurement_key(self, measurement_key: str):
        """set measurement_key"""
        self.general["measurement_key"] = measurement_key
        logger.info("(%s) measurement_key = %s", self._name, measurement_key)

    @property
    def specific_keys(self):
        """get specific_keys"""
        return self.general["specific_keys"]

    @specific_keys.setter
    def specific_keys(self, specific_keys: list[str]):
        """set specific_keys"""
        self.general["specific_keys"] = specific_keys
        logger.info("(%s) specific_keys = %s", self._name, specific_keys)

    @property
    def y_unsorted(self):
        """get y_unsorted"""
        return self.general["y_unsorted"]

    @y_unsorted.setter
    def y_unsorted(self, y_unsorted: np.ndarray):
        """set y_unsorted"""
        self.general["y_unsorted"] = y_unsorted
        logger.info("(%s) y_unsorted = %s", self._name, y_unsorted)

    @property
    def upsampling(self):
        """get upsampling"""
        return self.general["upsampling"]

    @upsampling.setter
    def upsampling(self, upsampling):
        """set upsampling"""
        self.general["upsampling"] = upsampling
        logger.info("(%s) upsampling = %s", self._name, upsampling)
