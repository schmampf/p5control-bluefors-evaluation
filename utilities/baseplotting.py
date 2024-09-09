"""
Description.

"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import savgol_filter

from utilities.baseevaluation import BaseEvaluation
from utilities.corporate_design_colors_v4 import cmap

logger = logging.getLogger(__name__)

PLOT_KEYS = {
    "y_axis": ["self.y_axis", r"$y$ (arb. u.)"],
    "V_bias_up_µV": [
        "self.voltage_axis*1e6",
        r"$V_\mathrm{Bias}^\rightarrow$ (µV)",
    ],
    "V_bias_up_V": [
        "self.voltage_axis*1e0",
        r"$V_\mathrm{Bias}^\rightarrow$ (V)",
    ],
    "V_bias_up_mV": [
        "self.mapped['voltage_axis']*1e3",
        r"$V_\mathrm{Bias}^\rightarrow$ (mV)",
    ],
    "V_bias_down_µV": [
        "self.voltage_axis*1e6",
        r"$V_\mathrm{Bias}^\leftarrow$ (µV)",
    ],
    "V_bias_down_mV": [
        "self.voltage_axis*1e3",
        r"$V_\mathrm{Bias}^\leftarrow$ (mV)",
    ],
    "V_bias_down_V": [
        "self.voltage_axis*1e0",
        r"$V_\mathrm{Bias}^\leftarrow$ (V)",
    ],
    "heater_power_µW": ["self.y_axis*1e6", r"$P_\mathrm{Heater}$ (µW)"],
    "heater_power_mW": ["self.y_axis*1e3", r"$P_\mathrm{Heater}$ (mW)"],
    "T_all_up_mK": ["self.temperature_all_up*1e3", r"$T_{Sample}$ (mK)"],
    "T_all_up_K": ["self.temperature_all_up*1e0", r"$T_{Sample}$ (K)"],
    "T_up_mK": ["self.temperature_mean_up*1e3", r"$T_\mathrm{Sample}$ (mK)"],
    "T_up_K": ["self.temperature_mean_up*1e0", r"$T_\mathrm{Sample}$ (K)"],
    "T_axis_up_K": [
        "self.mapped_over_temperature['temperature_axis']",
        r"$T_\mathrm{Sample}^\rightarrow$ (K)",
    ],
    "T_axis_down_K": [
        "self.temperature_axis",
        r"$T_\mathrm{Sample}^\leftarrow$ (K)",
    ],
    "dIdV_up": ["self.differential_conductance_up", r"d$I/$d$V$ ($G_0$)"],
    "dIdV_up_T": [
        "self.mapped_over_temperature['differential_conductance_up']",
        r"d$I/$d$V$ ($G_0$)",
    ],
    "dIdV_down": ["self.differential_conductance_down", r"d$I/$d$V$ ($G_0$)"],
    "dIdV_down_T": [
        "self.differential_conductance_down_over_temperature",
        r"d$I/$d$V$ ($G_0$)",
    ],
    "uH_up_mT": ["self.y_axis*1e3", r"$\mu_0H^\rightarrow$ (mT)"],
    "uH_up_T": ["self.y_axis", r"$\mu_0H^\rightarrow$ (T)"],
    "uH_down_mT": ["self.y_axis*1e3", r"$\mu_0H^\leftarrow$ (mT)"],
    "uH_down_T": ["self.y_axis", r"$\mu_0H^\leftarrow$ (T)"],
    "uH_mT": ["self.y_axis*1e3", r"$\mu_0H$ (mT)"],
    "uH_T": ["self.y_axis", r"$\mu_0H$ (T)"],
    "V_gate_up_V": ["self.y_axis", r"$V_\mathrm{Gate}^\rightarrow$ (V)"],
    "V_gate_down_V": ["self.y_axis", r"$V_\mathrm{Gate}^\leftarrow$ (V)"],
    "V_gate_V": ["self.y_axis", r"$V_\mathrm{Gate}$ (V)"],
    "V_gate_up_mV": ["self.y_axis*1e3", r"$V_\mathrm{Gate}^\rightarrow$ (mV)"],
    "V_gate_down_mV": ["self.y_axis*1e3", r"$V_\mathrm{Gate}^\leftarrow$ (mV)"],
    "V_gate_mV": ["self.y_axis*1e3", r"$V_\mathrm{Gate}$ (mV)"],
    "time_up": ["self.time_up", r"time"],
}


def plot_map(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_lim: list[float] = [-1.0, 1.0],
    y_lim: list[float] = [-1.0, 1.0],
    z_lim: list[float] = [0.0, 0.0],
    x_label: str = r"$x$-label",
    y_label: str = r"$y$-label",
    z_label: str = r"$z$-label",
    fig_nr: int = 0,
    color_map=cmap(color="seeblau", bad="gray"),
    display_dpi: int = 100,
    contrast: float = 1.0,
):
    """
    Docstring
    """

    if z.dtype == np.dtype("int32"):
        logger.warning("z is integer. Sure?")

    stepsize_x = np.abs(x[-1] - x[-2]) / 2
    stepsize_y = np.abs(y[-1] - y[-2]) / 2
    x_ind = [np.abs(x - x_lim[0]).argmin(), np.abs(x - x_lim[1]).argmin()]
    y_ind = [np.abs(y - y_lim[0]).argmin(), np.abs(y - y_lim[1]).argmin()]

    ext = np.array(
        [
            x[x_ind[0]] - stepsize_x,
            x[x_ind[1]] + stepsize_x,
            y[y_ind[0]] - stepsize_y,
            y[y_ind[1]] + stepsize_y,
        ],
        dtype="float64",
    )
    z = np.array(z[y_ind[0] : y_ind[1], x_ind[0] : x_ind[1]], dtype="float64")
    x = np.array(x[x_ind[0] : x_ind[1]], dtype="float64")
    y = np.array(y[y_ind[0] : y_ind[1]], dtype="float64")

    if z_lim == [0, 0]:
        z_lim = [
            float(np.nanmean(z) - np.nanstd(z) / contrast),
            float(np.nanmean(z) + np.nanstd(z) / contrast),
        ]

    if x_lim[0] >= x_lim[1] or y_lim[0] >= y_lim[1] or z_lim[0] >= z_lim[1]:
        logger.warning("First of xy_lim must be smaller than first one.")

    plt.close(fig_nr)
    fig, (ax_z, ax_c) = plt.subplots(
        num=fig_nr,
        ncols=2,
        figsize=(6, 4),
        dpi=display_dpi,
        gridspec_kw={"width_ratios": [5.8, 0.2]},
        constrained_layout=True,
    )

    im = ax_z.imshow(
        z,
        extent=ext,
        aspect="auto",
        origin="lower",
        clim=z_lim,
        cmap=color_map,
        interpolation="none",
    )
    ax_z.set_xlabel(x_label)
    ax_z.set_ylabel(y_label)
    ax_z.ticklabel_format(axis="both", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_z.tick_params(direction="in")

    _ = fig.colorbar(im, label=z_label, cax=ax_c)
    ax_c.tick_params(direction="in")
    _ = ax_z.set_xlim(ext[0], ext[1])
    _ = ax_z.set_ylim(ext[2], ext[3])

    plt.show()

    return fig, ax_z, ax_c, x, y, z, ext


class BasePlotting(BaseEvaluation):
    """
    Description
    """

    def __init__(
        self,
        name="base plot",
    ):
        """
        Description
        """
        super().__init__(name=name)

        self.base["fig_folder"] = "figures/"

        self.possible_plot_keys = PLOT_KEYS

        self.plot = {
            "smoothing": True,
            "window_length": 20,
            "polyorder": 2,
            "fig_nr_show_map": 0,
            "display_dpi": 100,
            "png_dpi": 600,
            "pdf_dpi": 600,
            "save_pdf": False,
            "color_map": cmap(color="seeblau", bad="gray"),
            "contrast": 1,
            "x_lim": [-1.0, 1.0],
            "y_lim": [-1.0, 1.0],
            "z_lim": [-1.0, 1.0],
            "x_key": "V_bias_up_mV",
            "y_key": "T_axis_up_K",
            "z_key": "dIdV_up_T",
        }

        self.show_map = {}
        logger.info("(%s) ... BasePlotting initialized.", self._name)

    def showMap(
        self,
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

        x_key = self.plot["x_key"]
        y_key = self.plot["y_key"]
        z_key = self.plot["z_key"]

        logger.info(
            "(%s) showMap('%s', '%s', '%s')",
            self._name,
            x_key,
            y_key,
            z_key,
        )

        x_lim = self.plot["x_lim"]
        y_lim = self.plot["y_lim"]
        z_lim = self.plot["z_lim"]

        warning = False

        try:
            plot_key_x = self.possible_plot_keys[x_key]
        except KeyError:
            logger.warning("(%s) x_key not found.", self._name)
            warning = True

        try:
            plot_key_y = self.possible_plot_keys[y_key]
        except KeyError:
            logger.warning("(%s) y_key not found.", self._name)
            warning = True

        try:
            plot_key_z = self.possible_plot_keys[z_key]
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

            if self.plot["smoothing"]:
                z_data = savgol_filter(
                    z_data,
                    window_length=self.plot["window_length"],
                    polyorder=self.plot["polyorder"],
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
            fig_nr=self.plot["fig_nr_show_map"],
            color_map=self.plot["color_map"],
            display_dpi=self.plot["display_dpi"],
            contrast=self.plot["contrast"],
        )

        title = ""
        if warning:
            title = "Hier könnte ihre Werbung stehen."
        elif self.base["title"] is not None:
            title = self.base["title"]
        else:
            title = self.general["measurement_key"]
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
            "fig_nr": self.plot["fig_nr_show_map"],
            "color_map": self.plot["color_map"],
            "display_dpi": self.plot["display_dpi"],
            "x_key": x_key,
            "y_key": y_key,
            "z_key": z_key,
            "contrast": self.plot["contrast"],
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
                color_map=self.show_map["color_map"],
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
            "(%s) saveFigure() to %s%s.png",
            self._name,
            self.base["fig_folder"],
            self.base["title"],
        )

        # Handle Title
        title = f"{self.base['title']}"

        # Handle data folder
        folder = os.path.join(
            os.getcwd(), self.base["fig_folder"], self.base["sub_folder"]
        )
        check = os.path.isdir(folder)
        if not check:
            os.makedirs(folder)

        # Save Everything
        name = os.path.join(folder, title)
        self.show_map["fig"].savefig(f"{name}.png", dpi=self.plot["png_dpi"])
        if self.plot["save_pdf"]:  # save as pdf
            logger.info(
                "(%s) saveFigure() to %s%s.pdf",
                self._name,
                self.base["fig_folder"],
                self.base["title"],
            )
            self.show_map["fig"].savefig(f"{name}.pdf", dpi=self.plot["pdf_dpi"])

    @property
    def x_key(self):
        """get x_key"""
        return self.plot["x_key"]

    @x_key.setter
    def x_key(self, x_key: str):
        """set x_key"""
        self.plot["x_key"] = x_key
        logger.info("(%s) x_key = %s", self._name, x_key)

    @property
    def y_key(self):
        """get y_key"""
        return self.plot["y_key"]

    @y_key.setter
    def y_key(self, y_key: str):
        """set y_key"""
        self.plot["y_key"] = y_key
        logger.info("(%s) y_key = %s", self._name, y_key)

    @property
    def z_key(self):
        """get z_key"""
        return self.plot["z_key"]

    @z_key.setter
    def z_key(self, z_key: str):
        """set z_key"""
        self.plot["z_key"] = z_key
        logger.info("(%s) z_key = %s", self._name, z_key)

    @property
    def x_lim(self):
        """get x_lim"""
        return self.plot["x_lim"]

    @x_lim.setter
    def x_lim(self, x_lim: list[float]):
        """set x_lim"""
        self.plot["x_lim"] = x_lim
        logger.info("(%s) x_lim = %s", self._name, x_lim)

    @property
    def y_lim(self):
        """get y_lim"""
        return self.plot["y_lim"]

    @y_lim.setter
    def y_lim(self, y_lim: list[float]):
        """set y_lim"""
        self.plot["y_lim"] = y_lim
        logger.info("(%s) y_lim = %s", self._name, y_lim)

    @property
    def z_lim(self):
        """get z_lim"""
        return self.plot["z_lim"]

    @z_lim.setter
    def z_lim(self, z_lim: list[float]):
        """set z_lim"""
        self.plot["z_lim"] = z_lim
        logger.info("(%s) z_lim = %s", self._name, z_lim)

    @property
    def smoothing(self):
        """get smoothing"""
        return self.plot["smoothing"]

    @smoothing.setter
    def smoothing(self, smoothing: bool):
        """set smoothing"""
        self.plot["smoothing"] = smoothing
        logger.info("(%s) smoothing = %s", self._name, smoothing)

    @property
    def window_length(self):
        """get window_length"""
        return self.plot["window_length"]

    @window_length.setter
    def window_length(self, window_length: int):
        """set window_length"""
        self.plot["window_length"] = window_length
        logger.info("(%s) window_length = %s", self._name, window_length)

    @property
    def polyorder(self):
        """get polyorder"""
        return self.plot["polyorder"]

    @polyorder.setter
    def polyorder(self, polyorder: int):
        """set polyorder"""
        self.plot["polyorder"] = polyorder
        logger.info("(%s) polyorder = %s", self._name, polyorder)

    @property
    def display_dpi(self):
        """get display_dpi"""
        return self.plot["display_dpi"]

    @display_dpi.setter
    def display_dpi(self, display_dpi: int):
        """set display_dpi"""
        self.plot["display_dpi"] = display_dpi
        logger.info("(%s) display_dpi = %s", self._name, display_dpi)

    @property
    def png_dpi(self):
        """get png_dpi"""
        return self.plot["png_dpi"]

    @png_dpi.setter
    def png_dpi(self, png_dpi: int):
        """set png_dpi"""
        self.plot["png_dpi"] = png_dpi
        logger.info("(%s) png_dpi = %s", self._name, png_dpi)

    @property
    def pdf_dpi(self):
        """get pdf_dpi"""
        return self.plot["pdf_dpi"]

    @pdf_dpi.setter
    def pdf_dpi(self, pdf_dpi: int):
        """set pdf_dpi"""
        self.plot["pdf_dpi"] = pdf_dpi
        logger.info("(%s) pdf_dpi = %s", self._name, pdf_dpi)

    @property
    def save_pdf(self):
        """get save_pdf"""
        return self.plot["save_pdf"]

    @save_pdf.setter
    def save_pdf(self, save_pdf: bool):
        """set save_pdf"""
        self.plot["save_pdf"] = save_pdf
        logger.info("(%s) save_pdf = %s", self._name, save_pdf)

    @property
    def color_map(self):
        """get color_map"""
        return self.plot["color_map"]

    @color_map.setter
    def color_map(self, color_map):
        """set color_map"""
        self.plot["color_map"] = color_map
        logger.info("(%s) color_map = %s", self._name, color_map)

    @property
    def contrast(self):
        """get contrast"""
        return self.plot["contrast"]

    @contrast.setter
    def contrast(self, contrast: float):
        """set contrast"""
        self.plot["contrast"] = contrast
        logger.info("(%s) contrast = %s", self._name, contrast)
