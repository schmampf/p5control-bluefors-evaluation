"""
Module a base evaluation, that evaluates data according to p5control-bluefors.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from h5py import File
from scipy import constants

from utilities.ivevaluation import IVEvaluation
from utilities.basefunctions import linear_fit
from utilities.basefunctions import bin_y_over_x
from utilities.basefunctions import bin_z_over_y

plt.ioff()

logger = logging.getLogger(__name__)


class TemperatureEvaluation(IVEvaluation):
    """
    Description
    """

    def __init__(
        self,
        name="temperature eva",
    ):
        """
        Description
        """
        super().__init__(name=name)

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

        if self.index_trigger_up is not None:
            (
                self.mapped_over_temperature["current_up"],
                self.mapped_over_temperature["counter_up"],
            ) = bin_z_over_y(
                self.mapped["temperature_up"],
                self.mapped["current_up"],
                self.mapped_over_temperature["temperature_axis"],
            )
            (
                self.mapped_over_temperature["differential_conductance_up"],
                _,
            ) = bin_z_over_y(
                self.mapped["temperature_up"],
                self.mapped["differential_conductance_up"],
                self.mapped_over_temperature["temperature_axis"],
            )
            self.mapped_over_temperature["y_axis_up"], _ = bin_y_over_x(
                self.mapped["temperature_up"],
                self.mapped["y_axis"],
                self.mapped_over_temperature["temperature_axis"],
            )

        if self.index_trigger_down is not None:
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


"""

from utilities.basefunctions import bin_z_over_y

            "amplitude_minimum": np.nan,
            "amplitude_maximum": np.nan,
            "amplitude_bins": np.nan,
            "amplitude_axis": np.array([]),
        self.setA(0, 0.1, amplitude_bins=100)

    def setA(
        self,
        amplitude_absolute=None,
        amplitude_minimum=None,
        amplitude_maximum=None,
        amplitude_bins=None,
    ):
        if amplitude_absolute is not None:
            amplitude_minimum = -amplitude_absolute
            amplitude_maximum = +amplitude_absolute

        if amplitude_minimum is not None:
            self.amplitude_minimum = amplitude_minimum
        if amplitude_maximum is not None:
            self.amplitude_maximum = amplitude_maximum
        if amplitude_bins is not None:
            self.amplitude_bins = amplitude_bins

        self.amplitude_axis = np.linspace(
            self.amplitude_minimum,
            self.amplitude_maximum,
            int(self.amplitude_bins) + 1,
        )

        logger.debug(
            "(%s) setA(%s, %s, %s)",
            self._name,
            self.amplitude_minimum,
            self.amplitude_maximum,
            self.amplitude_bins,
        )

    def get_amplitude_binning_done(self, new_dictionary, old_dictionary):
        # P = 10*np.log10(A**2*10)
        # A = np.sqrt((10**(P/10))/10)
        y_amplitude = np.sqrt((10 ** (self.mapped["y_axis"] / 10)) / 10)
        print(y_amplitude)
        if self.eva_current:
            (
                new_dictionary["current"],
                new_dictionary["counter"],
            ) = bin_z_over_y(
                y_amplitude,
                old_dictionary["current"],
                self.mapped["amplitude_axis"],
            )
            (
                new_dictionary["differential_conductance"],
                _,
            ) = bin_z_over_y(
                y_amplitude,
                old_dictionary["differential_conductance"],
                self.mapped["amplitude_axis"],
            )
            new_dictionary["temperature"], _ = bin_y_over_x(
                y_amplitude,
                old_dictionary["temperature"],
                self.mapped["amplitude_axis"],
            )

    def getMapsA(self):
        if self.index_trigger_up is not None:
            self.get_amplitude_binning_done(self.up_sweep_over_amplitude, self.up_sweep)"

    @property
    def amplitude_minimum(self):
        return self.mapped["amplitude_minimum"]

    @amplitude_minimum.setter
    def amplitude_minimum(self, amplitude_minimum):
        self.mapped["amplitude_minimum"] = amplitude_minimum

    @property
    def amplitude_maximum(self):
        return self.mapped["amplitude_maximum"]

    @amplitude_maximum.setter
    def amplitude_maximum(self, amplitude_maximum):
        self.mapped["amplitude_maximum"] = amplitude_maximum

    @property
    def amplitude_bins(self):
        return self.mapped["amplitude_bins"]

    @amplitude_bins.setter
    def amplitude_bins(self, amplitude_bins):
        self.mapped["amplitude_bins"] = amplitude_bins

    @property
    def amplitude_axis(self):
        return self.mapped["amplitude_axis"]

    @amplitude_axis.setter
    def amplitude_axis(self, amplitude_axis):
        self.mapped["amplitude_axis"] = amplitude_axis

"""
