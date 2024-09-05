"""
Description.

"""

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
    "V_bias_up_mV": [
        "self.voltage_axis*1e3",
        r"$V_\mathrm{Bias}^\rightarrow$ (mV)",
    ],
    "V_bias_up_V": [
        "self.voltage_axis*1e0",
        r"$V_\mathrm{Bias}^\rightarrow$ (V)",
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
        "self.temperature_axis",
        r"$T_\mathrm{Sample}^\rightarrow$ (K)",
    ],
    "T_axis_down_K": [
        "self.temperature_axis",
        r"$T_\mathrm{Sample}^\leftarrow$ (K)",
    ],
    "dIdV_up": ["self.differential_conductance_up", r"d$I/$d$V$ ($G_0$)"],
    "dIdV_up_T": [
        "self.differential_conductance_up_over_temperature",
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
    cmap=cmap(color="seeblau", bad="gray"),
    display_dpi: int = 100,
    contrast: float = 1.0,
):

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
        cmap=cmap,
        interpolation="none",
    )
    ax_z.set_xlabel(x_label)
    ax_z.set_ylabel(y_label)
    ax_z.ticklabel_format(axis="both", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_z.tick_params(direction="in")

    cbar = fig.colorbar(im, label=z_label, cax=ax_c)
    ax_c.tick_params(direction="in")
    lim = ax_z.set_xlim(ext[0], ext[1])
    lim = ax_z.set_ylim(ext[2], ext[3])

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
        }

        self.show_map = {
            "x_lim": [],
            "y_lim": [],
            "z_lim": [],
        }
        logger.info("(%s) ... BasePlotting initialized.", self._name)

    def showMap(
        self,
        x_key: str = "V_bias_up_mV",
        y_key: str = "y_axis",
        z_key: str = "dIdV_up",
        x_lim=None,
        y_lim=None,
        z_lim=None,
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
            x_lim = [-1.0, 1.0]
        if y_lim is None:
            y_lim = [-1.0, 1.0]
        if z_lim is None:
            z_lim = [-1.0, 1.0]

        logger.info(
            "(%s) showMap(%s, %s)",
            self._name,
            [x_key, y_key, z_key],
            [x_lim, y_lim, z_lim],
        )

        print(x_key, type(x_key))
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
            cmap=self.plot["color_map"],
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
            "cmap": self.plot["cmap"],
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
