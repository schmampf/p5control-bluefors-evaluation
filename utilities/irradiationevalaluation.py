"""
Module a base evaluation, that evaluates data according to p5control-bluefors.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from h5py import File
from scipy import constants

from tqdm.contrib.itertools import product

from utilities.baseevaluation import BaseEvaluation
from utilities.basefunctions import linear_fit
from utilities.basefunctions import bin_y_over_x

plt.ioff()

logger = logging.getLogger(__name__)


class IrradiationEvaluation(BaseEvaluation):
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

        self.irradiation = {
            "nu": np.array([]),
            "v_ac": np.array([]),
            "keys": np.array([]),
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

        logger.info("(%s) ... IrradiationEvaluation initialized.", self._name)

    def setMultipleKeys(
        self,
        i0: list[int] = [4, 14],
        i1: list[int] = [10, 19],
        norm: list[float] = [1e9, 1e0],
        zero: str = "no_irradiation",
        shape: list[int] = [31, 80],
    ):

        # Access File
        try:
            file_name = (
                self.base["file_directory"]
                + self.base["file_folder"]
                + self.base["file_name"]
            )
            data_file = File(file_name, "r")
            measurement_data = data_file.get(f"measurement/{self.measurement_key}")
        except AttributeError:
            logger.error("(%s) File can not be found!", self._name)
            return
        except KeyError:
            logger.error("(%s) Measurement can not be found!", self._name)
            return

        keys = list(measurement_data.keys())  # type: ignore

        keys.remove(zero)
        keys = np.array(keys, dtype="S30")
        nu = np.zeros((np.shape(keys)[0]), dtype=np.float64)
        v_ac = np.zeros((np.shape(keys)[0]), dtype=np.float64)

        for i, key in enumerate(keys):
            nu[i] = float(key[i0[0] : i1[0]]) * norm[0]
            v_ac[i] = float(key[i0[1] : i1[1]]) * norm[1]

        nu = nu.reshape(shape[0], shape[1])
        v_ac = v_ac.reshape(shape[0], shape[1])
        keys = keys.reshape(shape[0], shape[1])

        nu_0 = np.full((31, 1), 0.0)
        v_ac_0 = np.full((31, 1), 0.0)
        keys_0 = np.full((31, 1), zero, dtype="S30")

        self.irradiation["nu"] = np.concatenate((nu_0, nu), axis=1)
        self.irradiation["v_ac"] = np.concatenate((v_ac_0, v_ac), axis=1)
        self.irradiation["keys"] = np.concatenate((keys_0, keys), axis=1)

    def getIrradiationMaps(self):
        logger.info("(%s) getIrradiationMaps()", self._name)

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

        v_axis = self.mapped["voltage_axis"]
        self.irradiation["voltage_axis"] = v_axis
        nu = self.irradiation["nu"]
        v_ac = self.irradiation["v_ac"]

        keys = self.irradiation["keys"]

        len_voltage = np.shape(v_axis)[0]
        len_nu = np.shape(nu)[0]
        len_v_ac = np.shape(v_ac)[1]

        # Initialize all values
        self.irradiation["voltage_offset_1"] = np.full(
            (len_nu, len_v_ac), np.nan, dtype="float64"
        )
        self.irradiation["voltage_offset_2"] = np.full(
            (len_nu, len_v_ac), np.nan, dtype="float64"
        )

        if self.general["index_trigger_up"] is not None:
            self.irradiation["current_up"] = np.full(
                (len_nu, len_v_ac, len_voltage), np.nan, dtype="float64"
            )
            self.irradiation["time_up"] = np.full(
                (len_nu, len_v_ac, len_voltage), np.nan, dtype="float64"
            )
            self.irradiation["temperature_all_up"] = np.full(
                (len_nu, len_v_ac, len_voltage), np.nan, dtype="float64"
            )
            self.irradiation["time_up_start"] = np.full(
                (len_nu, len_v_ac), np.nan, dtype="float64"
            )
            self.irradiation["time_up_stop"] = np.full(
                (len_nu, len_v_ac), np.nan, dtype="float64"
            )

        if self.general["index_trigger_down"] is not None:
            self.irradiation["current_down"] = np.full(
                (len_nu, len_v_ac, len_voltage), np.nan, dtype="float64"
            )
            self.irradiation["time_down"] = np.full(
                (len_nu, len_v_ac, len_voltage), np.nan, dtype="float64"
            )
            self.irradiation["temperature_all_down"] = np.full(
                (len_nu, len_v_ac, len_voltage), np.nan, dtype="float64"
            )
            self.irradiation["time_down_start"] = np.full(
                (len_nu, len_v_ac), np.nan, dtype="float64"
            )
            self.irradiation["time_down_stop"] = np.full(
                (len_nu, len_v_ac), np.nan, dtype="float64"
            )

        for i, j in product(range(len_nu), range(len_v_ac)):
            key = keys[i, j].decode("utf-8")

            # Retrieve Offset Dataset
            measurement_data_offset = np.array(
                data_file.get(
                    f"measurement/{self.general['measurement_key']}/{key}/offset/adwin"
                )
            )
            # Calculate Offsets
            self.irradiation["voltage_offset_1"][i, j] = np.nanmean(
                np.array(measurement_data_offset["V1"])
            )
            self.irradiation["voltage_offset_2"][i, j] = np.nanmean(
                np.array(measurement_data_offset["V2"])
            )
            # Retrieve Sweep Dataset
            measurement_data_sweep = np.array(
                data_file.get(
                    f"measurement/{self.general['measurement_key']}/{key}/sweep/adwin"
                )
            )
            # Get Voltage Readings of Adwin
            trigger = np.array(measurement_data_sweep["trigger"], dtype="int")
            time = np.array(measurement_data_sweep["time"], dtype="float64")
            v1 = np.array(measurement_data_sweep["V1"], dtype="float64")
            v2 = np.array(measurement_data_sweep["V2"], dtype="float64")

            # Calculate V, I
            v_raw = (v1 - self.irradiation["voltage_offset_1"][i, j]) / self.general[
                "voltage_amplification_1"
            ]
            i_raw = (
                (v2 - self.irradiation["voltage_offset_2"][i, j])
                / self.general["voltage_amplification_2"]
                / self.general["reference_resistor"]
            )

            if self.general["index_trigger_up"] is not None:
                # Get upsweep
                v_raw_up = v_raw[trigger == self.general["index_trigger_up"]]
                i_raw_up = i_raw[trigger == self.general["index_trigger_up"]]
                time_up = time[trigger == self.general["index_trigger_up"]]
                # Calculate Timepoints
                self.irradiation["time_up_start"][i, j] = time_up[0]
                self.irradiation["time_up_stop"][i, j] = time_up[-1]
                # Bin that stuff
                i_up, _ = bin_y_over_x(
                    v_raw_up,
                    i_raw_up,
                    v_axis,
                    upsampling=self.upsampling,
                )
                time_up, _ = bin_y_over_x(
                    v_raw_up,
                    time_up,
                    v_axis,
                    upsampling=self.upsampling,
                )
                # Save to Array
                self.irradiation["current_up"][i, j, :] = i_up
                self.irradiation["time_up"][i, j, :] = time_up

            if self.general["index_trigger_down"] is not None:
                # Get dwonsweep
                v_raw_down = v_raw[trigger == self.general["index_trigger_down"]]
                i_raw_down = i_raw[trigger == self.general["index_trigger_down"]]
                time_down = time[trigger == self.general["index_trigger_down"]]
                # Calculate Timepoints
                self.irradiation["time_down_start"][i, j] = time_down[0]
                self.irradiation["time_down_stop"][i, j] = time_down[-1]
                # Bin that stuff
                i_down, _ = bin_y_over_x(
                    v_raw_down,
                    i_raw_down,
                    v_axis,
                    upsampling=self.upsampling,
                )
                time_down, _ = bin_y_over_x(
                    v_raw_down,
                    time_down,
                    v_axis,
                    upsampling=self.upsampling,
                )
                # Save to Array
                self.irradiation["current_down"][i, j, :] = i_down
                self.irradiation["time_down"][i, j, :] = time_down

            # Retrieve Temperature Dataset
            data_set = data_file.get(
                f"measurement/{self.general['measurement_key']}/{key}/sweep"
            )
            if "bluefors" in data_set.keys():  # type: ignore
                measurement_data_temperature = np.array(
                    data_file.get(
                        f"measurement/{self.general['measurement_key']}/{key}/sweep/bluefors"
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
                    self.irradiation["temperature_all_up"][i, j, :] = temperature_up

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
                    self.irradiation["temperature_all_down"][i, j, :] = temperature_down
            else:
                measurement_data_temperature = False
                logger.error("(%s) No temperature data available!", self._name)

        conductance_quantum = constants.physical_constants["conductance quantum"][0]

        if self.general["index_trigger_up"] is not None:
            # calculating differential conductance
            self.irradiation["differential_conductance_up"] = (
                np.gradient(
                    self.irradiation["current_up"],
                    v_axis,
                    axis=2,
                )
                / conductance_quantum
            )
            self.irradiation["temperature_mean_up"] = np.nanmean(
                self.irradiation["temperature_all_up"], axis=2
            )

        if self.general["index_trigger_down"] is not None:
            self.irradiation["differential_conductance_down"] = (
                np.gradient(
                    self.irradiation["current_down"],
                    v_axis,
                    axis=2,
                )
                / conductance_quantum
            )
            self.irradiation["temperature_mean_down"] = np.nanmean(
                self.irradiation["temperature_all_down"], axis=2
            )
