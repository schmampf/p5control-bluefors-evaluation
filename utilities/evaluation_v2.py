"""Module providing a function printing python version."""

import os
import platform
import logging
import pickle

from importlib import reload

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from h5py import File
from PIL import Image
from scipy import constants
from scipy.signal import savgol_filter

from .corporate_design_colors_v4 import cmap
from .evaluation_helper_v2 import linfit
from .evaluation_helper_v2 import bin_y_over_x
from .evaluation_helper_v2 import bin_z_over_y
from .evaluation_helper_v2 import plot_map
from .evaluation_helper_v2 import PLOT_KEYS

reload(logging)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class EvaluationScript:
    """class doc string."""

    def __init__(
        self,
        name="eva",
    ):
        self._name = name

        match platform.system():
            case "Darwin":
                self.file_directory = "/Users/oliver/Documents/measurement_data/"
            case "Linux":
                self.file_directory = "/home/oliver/Documents/measurement_data/"
            case default:
                self.file_directory = ""
                logger.warning(
                    "(%s) needs a file directory under %s.",
                    self._name,
                    default,
                )

        # where to find the measurement file
        self.file_name = ""
        self.file_folder = ""

        # where to save fiure and data
        self.sub_folder = ""
        self.fig_folder = "figures/"
        self.data_folder = "data/"

        # initialize key lists
        self.measurement_key = ""
        self.specific_keys = []
        self.possible_measurement_keys = {
            "temperatures": [7, -3, 1e-6, "no_heater"],
            "temperatures_up": [7, -2, 1e-6, "no_heater"],
            "gate_voltages": [5, -2, 1e-3, "no_gate"],
        }

        # initialize values for calculation
        self.voltage_amplification_1 = 1
        self.voltage_amplification_2 = 1
        self.reference_resistor = 51.689e3  # Ohm

        # initialize trigger values
        self.trigger_up = 1
        self.trigger_down = 2

        # initilize voltage axis
        self.voltage_minimum = -1.8e-3
        self.voltage_maximum = +1.8e-3
        self.voltage_bins = 900
        self.voltage_axis = np.linspace(
            self.voltage_minimum,
            self.voltage_maximum,
            int(self.voltage_bins) + 1,
        )

        # initialize temperature axis
        self.temperature_minimum = 0
        self.temperature_maximum = 2
        self.temperature_bins = 2000
        self.temperature_axis = np.linspace(
            self.temperature_minimum,
            self.temperature_maximum,
            int(self.temperature_bins) + 1,
        )

        # initialize smoothing values
        self.upsampling = None
        self.window_length = 20
        self.polyorder = 2

        # initialize plotting values
        self.title = ""
        self.fig_nr_show_map = 0
        self.display_dpi = 100
        self.png_dpi = 600
        self.pdf_dpi = 600
        self.pdf = False
        self.cmap = cmap(color="seeblau", bad="gray")
        self.contrast = 1

        self.x_lim = [-1.0, 1.0]
        self.y_lim = [-1.0, 1.0]
        self.z_lim = [-1.0, 1.0]

        self.plot_keys = PLOT_KEYS
        self.show_map = {}

        # initialize saving values
        self.ignore_while_saving = []

        # initialize calculation values
        self.y_unsorted = np.array([])
        self.y_axis = np.array([])

        self.voltage_offset_1 = np.array([])
        self.voltage_offset_2 = np.array([])
        self.current_up = np.array([])
        self.current_down = np.array([])
        self.differential_conductance_up = np.array([])
        self.differential_conductance_down = np.array([])

        self.temperature_all_up = np.array([])
        self.temperature_all_down = np.array([])
        self.temperature_mean_up = np.array([])
        self.temperature_mean_down = np.array([])

        self.time_up = np.array([])
        self.time_down = np.array([])
        self.time_up_start = np.array([])
        self.time_up_stop = np.array([])
        self.time_down_start = np.array([])
        self.time_down_stop = np.array([])

        self.counter_up = np.array([])
        self.counter_down = np.array([])
        self.current_up_over_temperature = np.array([])
        self.current_down_over_temperature = np.array([])
        self.differential_conductance_up_over_temperature = np.array([])
        self.differential_conductance_down_over_temperature = np.array([])
        self.y_axis_up_over_temperature = np.array([])
        self.y_axis_down_over_temperature = np.array([])

        logger.info("(%s) ... initialized.", self._name)

    def showAmplifications(
        self,
    ):
        """
        Shows amplification over time during whole measurement.
        """
        logger.info("(%s) showAmplifications()", self._name)

        data_file = File(
            f"{self.file_directory}{self.file_folder}{self.file_name}", "r"
        )
        femto_data = np.array(data_file.get("status/femto"))
        time = femto_data["time"]
        amplification_a = femto_data["amp_A"]
        amplification_b = femto_data["amp_B"]
        plt.figure(1000, figsize=(6, 1))
        plt.semilogy(time, amplification_a, "-", label="voltage amplification 1")
        plt.semilogy(time, amplification_b, "--", label="voltage amplification 2")
        plt.legend()
        plt.title("Femto Amplifications according to Status")
        plt.xlabel("time (s)")
        plt.ylabel("amplification")
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
        self.voltage_amplification_1 = voltage_amplification_1
        self.voltage_amplification_2 = voltage_amplification_2

    def showMeasurements(self):
        """
        Shows available Measurements in File.
        """
        logger.info("(%s) showMeasurements()", self._name)

        data_file = File(
            f"{self.file_directory}{self.file_folder}{self.file_name}", "r"
        )
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
            data_file = File(
                f"{self.file_directory}{self.file_folder}{self.file_name}", "r"
            )
            measurement_data = data_file.get(f"measurement/{measurement_key}")
            self.specific_keys = list(measurement_data)  # type: ignore
            self.measurement_key = measurement_key
        except KeyError:
            self.specific_keys = []
            self.measurement_key = ""
            logger.error("(%s) '%s' found in File.", self._name, measurement_key)

    def showKeys(self):
        """
        Shows available Keys in Measurement.
        """
        logger.info("(%s) showKeys()", self._name)
        show_keys = self.specific_keys[:2] + self.specific_keys[-2:]
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
        if self.measurement_key == "":
            logger.warning("(%s) Do setMeasurement() first.", self._name)
            return

        if parameters is None:
            parameters = self.possible_measurement_keys[self.measurement_key]

        logger.info("(%s) setKeys(%s)", self._name, parameters)

        try:
            i0 = parameters[0]
            i1 = parameters[1]
            norm = parameters[2]
            to_pop = parameters[3]
        except IndexError:
            logger.warning("(%s) List of Parameter is incompete.", self._name)
            return

        if to_pop in self.specific_keys:
            self.specific_keys.remove(to_pop)
        else:
            logger.warning("(%s) Key to pop is not found.", self._name)

        y = []
        for key in self.specific_keys:
            temp = key[i0:i1]
            temp = float(temp) * norm
            y.append(temp)
        y = np.array(y)

        self.y_unsorted = y

    def setV(
        self,
        voltage_absolute: float = np.nan,
        voltage_minimum: float = np.nan,
        voltage_maximum: float = np.nan,
        voltage_bins: float = np.nan,
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
        if not np.isnan(voltage_absolute):
            voltage_minimum = -voltage_absolute
            voltage_maximum = +voltage_absolute
        if not np.isnan(voltage_minimum):
            self.voltage_minimum = voltage_minimum
        if not np.isnan(voltage_maximum):
            self.voltage_maximum = voltage_maximum
        if not np.isnan(voltage_bins):
            self.voltage_bins = voltage_bins

        logger.info(
            "(%s) setV(%s, %s, %s)",
            self._name,
            self.voltage_minimum,
            self.voltage_maximum,
            self.voltage_bins,
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

        # Calculate new V-Axis
        self.voltage_axis = np.linspace(
            self.voltage_minimum,
            self.voltage_maximum,
            int(self.voltage_bins) + 1,
        )

        # Access File
        try:
            data_file = File(
                f"{self.file_directory}{self.file_folder}{self.file_name}", "r"
            )
        except AttributeError:
            logger.error("(%s) File can not be found!", self._name)
            return
        except KeyError:
            logger.error("(%s) Measurement can not be found!", self._name)
            return

        len_voltage = np.shape(self.voltage_axis)[0]
        len_y = np.shape(self.y_unsorted)[0]

        # Initialize all values
        self.current_up = np.full((len_y, len_voltage), np.nan, dtype="float64")
        self.current_down = np.full((len_y, len_voltage), np.nan, dtype="float64")
        self.time_up = np.full((len_y, len_voltage), np.nan, dtype="float64")
        self.time_down = np.full((len_y, len_voltage), np.nan, dtype="float64")
        self.temperature_all_up = np.full((len_y, len_voltage), np.nan, dtype="float64")
        self.temperature_all_down = np.full(
            (len_y, len_voltage), np.nan, dtype="float64"
        )

        self.time_up_start = np.full(len_y, np.nan, dtype="float64")
        self.time_up_stop = np.full(len_y, np.nan, dtype="float64")
        self.time_down_start = np.full(len_y, np.nan, dtype="float64")
        self.time_down_stop = np.full(len_y, np.nan, dtype="float64")
        self.voltage_offset_1 = np.full(len_y, np.nan, dtype="float64")
        self.voltage_offset_2 = np.full(len_y, np.nan, dtype="float64")

        # Iterate over Keys
        for i, k in enumerate(tqdm(self.specific_keys)):

            # Retrieve Offset Dataset
            measurement_data_offset = np.array(
                data_file.get(f"measurement/{self.measurement_key}/{k}/offset/adwin")
            )
            # Calculate Offsets
            self.voltage_offset_1[i] = np.nanmean(
                np.array(measurement_data_offset["V1"])
            )
            self.voltage_offset_2[i] = np.nanmean(
                np.array(measurement_data_offset["V2"])
            )

            # Retrieve Sweep Dataset
            measurement_data_sweep = np.array(
                data_file.get(f"measurement/{self.measurement_key}/{k}/sweep/adwin")
            )
            # Get Voltage Readings of Adwin
            trigger = np.array(measurement_data_sweep["trigger"], dtype="int")
            time = np.array(measurement_data_sweep["time"], dtype="float64")
            v1 = np.array(measurement_data_sweep["V1"], dtype="float64")
            v2 = np.array(measurement_data_sweep["V2"], dtype="float64")

            # Calculate V, I
            v_raw = (v1 - self.voltage_offset_1[i]) / self.voltage_amplification_1
            i_raw = (
                (v2 - self.voltage_offset_2[i])
                / self.voltage_amplification_2
                / self.reference_resistor
            )

            if self.trigger_up is not None:
                # Get upsweep
                v_raw_up = v_raw[trigger == self.trigger_up]
                i_raw_up = i_raw[trigger == self.trigger_up]
                time_up = time[trigger == self.trigger_up]
                # Calculate Timepoints
                self.time_up_start[i] = time_up[0]
                self.time_up_stop[i] = time_up[-1]
                # Bin that stuff
                i_up, _ = bin_y_over_x(v_raw_up, i_raw_up, self.voltage_axis)
                time_up, _ = bin_y_over_x(v_raw_up, time_up, self.voltage_axis)
                # Save to Array
                self.current_up[i, :] = i_up
                self.time_up[i, :] = time_up

            if self.trigger_down is not None:
                # Get dwonsweep
                v_raw_down = v_raw[trigger == self.trigger_down]
                i_raw_down = i_raw[trigger == self.trigger_down]
                time_down = time[trigger == self.trigger_down]
                # Calculate Timepoints
                self.time_down_start[i] = time_down[0]
                self.time_down_stop[i] = time_down[-1]
                # Bin that stuff
                i_down, _ = bin_y_over_x(v_raw_down, i_raw_down, self.voltage_axis)
                time_down, _ = bin_y_over_x(v_raw_down, time_down, self.voltage_axis)
                # Save to Array
                self.current_down[i, :] = i_down
                self.time_down[i, :] = time_down

            # Retrieve Temperature Dataset
            data_set = data_file.get(f"measurement/{self.measurement_key}/{k}/sweep")
            if "bluefors" in data_set.keys():  # type: ignore
                measurement_data_temperature = np.array(
                    data_file.get(
                        f"measurement/{self.measurement_key}/{k}/sweep/bluefors"
                    )
                )
                temporary_time = measurement_data_temperature["time"]
                temporary_temperature = measurement_data_temperature["Tsample"]

                if self.trigger_up is not None:
                    temporary_time_up = linfit(time_up)
                    if temporary_time_up[0] > temporary_time_up[1]:
                        temporary_time_up = np.flip(temporary_time_up)
                    temperature_up, _ = bin_y_over_x(
                        temporary_time,
                        temporary_temperature,
                        temporary_time_up,
                        upsampling=1000,
                    )
                    self.temperature_all_up[i, :] = temperature_up

                if self.trigger_down is not None:
                    temporary_time_down = linfit(time_down)
                    if temporary_time_down[0] > temporary_time_down[1]:
                        temporary_time_down = np.flip(temporary_time_down)
                    temperature_down, _ = bin_y_over_x(
                        temporary_time,
                        temporary_temperature,
                        temporary_time_down,
                        upsampling=1000,
                    )
                    self.temperature_all_down[i, :] = temperature_down
            else:
                measurement_data_temperature = False
                logger.error("(%s) No temperature data available!", self._name)

        # sorting afterwards, because of probably unknown characters in keys
        indices = np.argsort(self.y_unsorted)
        self.y_axis = self.y_unsorted[indices]
        self.voltage_offset_1 = self.voltage_offset_1[indices]
        self.voltage_offset_2 = self.voltage_offset_2[indices]

        if self.trigger_up is not None:
            self.current_up = self.current_up[indices, :]
            self.time_up = self.time_up[indices, :]
            self.temperature_all_up = self.temperature_all_up[indices, :]
            self.time_up_start = self.time_up_start[indices]
            self.time_up_stop = self.time_up_stop[indices]

        if self.trigger_down is not None:
            self.current_down = self.current_down[indices, :]
            self.time_down = self.time_down[indices, :]
            self.temperature_all_down = self.temperature_all_down[indices, :]
            self.time_down_start = self.time_down_start[indices]
            self.time_down_stop = self.time_down_stop[indices]

        if bounds is not None:
            self.y_axis = self.y_axis[bounds[0] : bounds[1]]
            self.voltage_offset_1 = self.voltage_offset_1[bounds[0] : bounds[1]]
            self.voltage_offset_2 = self.voltage_offset_2[bounds[0] : bounds[1]]

            if self.trigger_up is not None:
                self.current_up = self.current_up[bounds[0] : bounds[1], :]
                self.time_up = self.time_up[bounds[0] : bounds[1], :]
                self.temperature_all_up = self.temperature_all_up[
                    bounds[0] : bounds[1], :
                ]
                self.time_up_start = self.time_up_start[bounds[0] : bounds[1]]
                self.time_up_stop = self.time_up_stop[bounds[0] : bounds[1]]

            if self.trigger_down is not None:
                self.current_down = self.current_down[bounds[0] : bounds[1], :]
                self.time_down = self.time_down[bounds[0] : bounds[1], :]
                self.temperature_all_down = self.temperature_all_down[
                    bounds[0] : bounds[1], :
                ]
                self.time_down_start = self.time_down_start[bounds[0] : bounds[1]]
                self.time_down_stop = self.time_down_stop[bounds[0] : bounds[1]]

        conductance_quantum = constants.physical_constants["conductance quantum"][0]

        if self.trigger_up is not None:
            # calculating differential conductance
            self.differential_conductance_up = (
                np.gradient(self.current_up, self.voltage_axis, axis=1)
                / conductance_quantum
            )
            # calculates self.temperature_mean_up, self.temperature_mean_down
            self.temperature_mean_up = np.nanmean(self.temperature_all_up, axis=1)

        if self.trigger_down is not None:
            self.differential_conductance_down = (
                np.gradient(self.current_down, self.voltage_axis, axis=1)
                / conductance_quantum
            )
            self.temperature_mean_down = np.nanmean(self.temperature_all_down, axis=1)

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
            self.temperature_minimum = temperature_minimum
        if not np.isnan(temperature_maximum):
            self.temperature_maximum = temperature_maximum
        if not np.isnan(temperature_bins):
            self.temperature_bins = temperature_bins

        logger.info(
            "(%s) setT(%s, %s, %s)",
            self._name,
            self.temperature_minimum,
            self.temperature_maximum,
            self.temperature_bins,
        )

    def getMapsT(
        self,
    ):
        """getMapsT()
        - Maps I, differential_conductance over linear T-axis
        """
        logger.info("(%s) getMapsT()", self._name)

        # Calculate new Temperature-Axis
        self.temperature_axis = np.linspace(
            self.temperature_minimum,
            self.temperature_maximum,
            int(self.temperature_bins) + 1,
        )

        self.current_up_over_temperature, self.counter_up = bin_z_over_y(
            self.temperature_mean_up, self.current_up, self.temperature_axis
        )
        self.current_down_over_temperature, self.counter_down = bin_z_over_y(
            self.temperature_mean_down, self.current_down, self.temperature_axis
        )
        self.differential_conductance_up_over_temperature, _ = bin_z_over_y(
            self.temperature_mean_up,
            self.differential_conductance_up,
            self.temperature_axis,
        )
        self.differential_conductance_down_over_temperature, _ = bin_z_over_y(
            self.temperature_mean_down,
            self.differential_conductance_down,
            self.temperature_axis,
        )
        self.y_axis_up_over_temperature, _ = bin_y_over_x(
            self.temperature_mean_up, self.y_axis, self.temperature_axis
        )
        self.y_axis_down_over_temperature, _ = bin_y_over_x(
            self.temperature_mean_down, self.y_axis, self.temperature_axis
        )

    def showMap(
        self,
        x_key: str = "V_bias_up_mV",
        y_key: str = "y_axis",
        z_key: str = "dIdV_up",
        x_lim=None,
        y_lim=None,
        z_lim=None,
        smoothing: bool = False,
        window_length: float = np.nan,
        polyorder: float = np.nan,
    ):
        """showMap()
        - checks for synthax errors
        - get data and label from plot_keys
        - calls plot_map()

        Parameters
        ----------
        x_key : str = 'V_bias_up_mV'
            select plot_key_x from self.plot_keys
        y_key : str = 'y_axis'
            select plot_key_y from self.plot_keys
        z_key : str = 'dIdV_up'
            select plot_key_z from self.plot_keys
        x_lim : list = [np.nan, np.nan]
            sets limits on x-Axis
        y_lim : list = [np.nan, np.nan]
            sets limits on y-Axis
        z_lim : list = [np.nan, np.nan]
            sets limits on z-Axis / colorbar
        """

        if x_lim is None:
            x_lim = self.x_lim
        if y_lim is None:
            y_lim = self.y_lim
        if z_lim is None:
            z_lim = self.z_lim

        if np.isnan(window_length):
            window_length = self.window_length
        else:
            self.window_length = window_length

        if np.isnan(polyorder):
            polyorder = self.polyorder
        else:
            self.polyorder = polyorder

        logger.info(
            "(%s) showMap(%s, %s, %s)",
            self._name,
            [x_key, y_key, z_key],
            [x_lim, y_lim, z_lim],
            [smoothing, window_length, polyorder],
        )

        warning = False

        try:
            plot_key_x = self.plot_keys[x_key]
        except KeyError:
            logger.warning("(%s) x_key not found.", self._name)
            warning = True

        try:
            plot_key_y = self.plot_keys[y_key]
        except KeyError:
            logger.warning("(%s) y_key not found.", self._name)
            warning = True

        try:
            plot_key_z = self.plot_keys[z_key]
        except KeyError:
            logger.warning("(%s) z_key not found.", self._name)
            warning = True

        if x_lim[0] >= x_lim[1]:
            logger.warning("(%s) x_lim = [lower_limit, upper_limit].", self._name)
            warning = True

        if y_lim[0] >= y_lim[1]:
            logger.warning("(%s) y_lim = [lower_limit, upper_limit].", self._name)
            warning = True

        if z_lim[0] >= z_lim[1]:
            logger.warning("(%s) z_lim = [lower_limit, upper_limit].", self._name)
            warning = True

        if not warning:
            try:
                x_data = eval(plot_key_x[0])  # pylint: disable=eval-used
                y_data = eval(plot_key_y[0])  # pylint: disable=eval-used
                z_data = eval(plot_key_z[0])  # pylint: disable=eval-used
            except AttributeError:
                logger.warning(
                    "(%s) Required data not found. Check if data is calculated and plot_keys!",
                    self._name,
                )
                return

            if smoothing:
                z_data = savgol_filter(
                    z_data, window_length=window_length, polyorder=polyorder
                )

            x_label = plot_key_x[1]
            y_label = plot_key_y[1]
            z_label = plot_key_z[1]

        else:
            logger.warning("(%s) Check Parameter!", self._name)
            try:
                image = Image.open(
                    "/home/oliver/Documents/p5control-bluefors-evaluation/"
                    + "utilities/blueforslogo.png",
                    mode="r",
                )
            except FileNotFoundError:
                logger.warning("(%s) Trick verreckt :/", self._name)
                return
            image = np.asarray(image, dtype="float64")
            z_data = np.flip(image[:, :, 1], axis=0)
            z_data[z_data >= 80] = 0.8
            z_data /= np.max(z_data)
            x_data = np.arange(image.shape[1])
            y_data = np.arange(image.shape[0])
            x_label = r"$x_\mathrm{}$ (pxl)"
            y_label = r"$y_\mathrm{}$ (pxl)"
            z_label = r"BlueFors (arb. u.)"
            x_lim = [0.0, 2000.0]
            y_lim = [0.0, 1000.0]
            z_lim = [0.0, 1.0]

        fig, ax_z, ax_c, x, y, z, ext = plot_map(
            x=x_data,
            y=y_data,
            z=z_data,
            x_lim=x_lim,
            y_lim=y_lim,
            z_lim=z_lim,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            fig_nr=self.fig_nr_show_map,
            cmap=self.cmap,
            display_dpi=self.display_dpi,
            contrast=self.contrast,
        )

        title = ""
        if warning:
            title = "Hier könnte ihre Werbung stehen."
        elif self.title is not None:
            title = self.title
        else:
            title = self.measurement_key
        plt.suptitle(title)

        self.show_map = {
            "title": title,
            "fig": fig,
            "ax_z": ax_z,
            "ax_c": ax_c,
            "ext": ext,
            "x": x,
            "y": y,
            "z": z,
            "x_data": x_data,
            "y_data": y_data,
            "z_data": z_data,
            "x_lim": x_lim,
            "y_lim": y_lim,
            "z_lim": z_lim,
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
            "fig_nr": self.fig_nr_show_map,
            "cmap": self.cmap,
            "display_dpi": self.display_dpi,
            "x_key": x_key,
            "y_key": y_key,
            "z_key": z_key,
            "contrast": self.contrast,
        }

    def reshowMap(
        self,
    ):
        """reshowMap()
        - shows Figure
        """
        logger.info(
            "(%s) reshowMap()",
            self._name,
        )
        if self.show_map:
            plot_map(
                x=self.show_map["x_data"],
                y=self.show_map["y_data"],
                z=self.show_map["z_data"],
                x_lim=self.show_map["x_lim"],
                y_lim=self.show_map["y_lim"],
                z_lim=self.show_map["z_lim"],
                x_label=self.show_map["x_label"],
                y_label=self.show_map["y_label"],
                z_label=self.show_map["z_label"],
                fig_nr=self.show_map["fig_nr"],
                cmap=self.show_map["cmap"],
                display_dpi=self.show_map["display_dpi"],
                contrast=self.show_map["contrast"],
            )
            plt.suptitle(self.show_map["title"])

    def saveFigure(
        self,
    ):
        """saveFigure()
        - safes Figure to self.fig_folder/self.title
        """
        logger.info(
            "(%s) saveFigure() to %s%s.png", self._name, self.fig_folder, self.title
        )

        # Handle Title
        title = f"{self.title}"

        # Handle data folder
        folder = os.path.join(os.getcwd(), self.fig_folder, self.sub_folder)
        check = os.path.isdir(folder)
        if not check:
            os.makedirs(folder)

        # Save Everything
        name = os.path.join(folder, title)
        self.show_map["fig"].savefig(f"{name}.png", dpi=self.png_dpi)
        if self.pdf:  # save as pdf
            logger.info(
                "(%s) saveFigure() to %s%s.pdf", self._name, self.fig_folder, self.title
            )
            self.show_map["show_map"].savefig(f"{name}.pdf", dpi=self.pdf_dpi)

    def saveData(
        self,
        title=None,
    ):
        """saveData()
        - safes self.__dict__ to pickle
        """
        logger.info("(%s) saveData()", self._name)

        # Handle Title
        if title is None:
            title = f"{self.title}.pickle"

        # Handle data folder
        folder = os.path.join(os.getcwd(), self.data_folder, self.sub_folder)
        check = os.path.isdir(folder)
        if not check and self.data_folder != "":
            os.makedirs(folder)

        # Get Dictionary
        data = {}
        for key, value in self.__dict__.items():
            if key not in self.ignore_while_saving:
                data[key] = value

        # save data to pickle
        name = os.path.join(os.getcwd(), self.data_folder, self.sub_folder, title)
        with open(name, "wb") as file:
            pickle.dump(data, file)

    def loadData(
        self,
        title=None,
    ):
        """loadData()
        - loads self.__dict__ from pickle
        """
        logger.info("(%s) loadData()", self._name)

        # Handle Title
        if title is None:
            title = f"{self.title}.pickle"

        # get data from pickle
        name = os.path.join(os.getcwd(), self.data_folder, self.sub_folder, title)
        with open(name, "rb") as file:
            data = pickle.load(file)

        # Save Data to self.
        for key, value in data.items():
            self.__dict__[key] = value

    def showData(
        self,
    ):
        """showData()
        - shows self.__dict__ from pickle
        """
        logger.info("(%s) showData()", self._name)

        # Get Dictionary
        data = {}
        for key, value in self.__dict__.items():
            if key not in self.ignore_while_saving:
                data[key] = value
        return data
