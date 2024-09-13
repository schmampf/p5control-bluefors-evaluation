"""
Description.

"""

import logging
import platform

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from utilities.corporate_design_colors_v4 import cmap

logger = logging.getLogger(__name__)

PLOT_KEYS = {
    "y_axis": ["self.mapped['y_axis']", r"$y$ (arb. u.)"],
    "V_bias_up_µV": [
        "self.mapped['voltage_axis']*1e6",
        r"$V_\mathrm{Bias}^\rightarrow$ (µV)",
    ],
    "V_bias_up_V": [
        "self.mapped['voltage_axis']*1e0",
        r"$V_\mathrm{Bias}^\rightarrow$ (V)",
    ],
    "V_bias_up_mV": [
        "self.mapped['voltage_axis']*1e3",
        r"$V_\mathrm{Bias}^\rightarrow$ (mV)",
    ],
    "V_bias_down_µV": [
        "self.mapped['voltage_axis']*1e6",
        r"$V_\mathrm{Bias}^\leftarrow$ (µV)",
    ],
    "V_bias_down_mV": [
        "self.mapped['voltage_axis']*1e3",
        r"$V_\mathrm{Bias}^\leftarrow$ (mV)",
    ],
    "V_bias_down_V": [
        "self.mapped['voltage_axis']*1e0",
        r"$V_\mathrm{Bias}^\leftarrow$ (V)",
    ],
    "dIdV_up": ["self.mapped['differential_conductance_up']", r"d$I/$d$V$ ($G_0$)"],
    "dIdV_up_T": [
        "self.mapped_over_temperature['differential_conductance_up']",
        r"d$I/$d$V$ ($G_0$)",
    ],
    "dIdV_down": ["self.mapped['differential_conductance_down']", r"d$I/$d$V$ ($G_0$)"],
    "dIdV_down_T": [
        "self.differential_conductance_down_over_temperature",
        r"d$I/$d$V$ ($G_0$)",
    ],
    "I_up_A": ["self.mapped['current_up']*1e0", r"$I$ (A)"],
    "I_up_mA": ["self.mapped['current_up']*1e3", r"$I$ (mA)"],
    "I_up_muA": ["self.mapped['current_up']*1e6", r"$I$ (µA)"],
    "I_up_nA": ["self.mapped['current_up']*1e9", r"$I$ (nA)"],
    "I_up_pA": ["self.mapped['current_up']*1e12", r"$I$ (pA)"],
    "I_down_A": ["self.mapped['current_down']*1e0", r"$I$ (A)"],
    "I_down_mA": ["self.mapped['current_down']*1e3", r"$I$ (mA)"],
    "I_down_muA": ["self.mapped['current_down']*1e6", r"$I$ (µA)"],
    "I_down_nA": ["self.mapped['current_down']*1e9", r"$I$ (nA)"],
    "I_down_pA": ["self.mapped['current_down']*1e12", r"$I$ (pA)"],
    "heater_power_µW": ["self.mapped['y_axis']*1e6", r"$P_\mathrm{Heater}$ (µW)"],
    "heater_power_mW": ["self.mapped['y_axis']*1e3", r"$P_\mathrm{Heater}$ (mW)"],
    "T_up_mK": ["self.mapped['temperature_mean_up']*1e3", r"$T_\mathrm{Sample}$ (mK)"],
    "T_up_K": ["self.mapped['temperature_mean_up']*1e0", r"$T_\mathrm{Sample}$ (K)"],
    "T_axis_up_K": [
        "self.mapped_over_temperature['temperature_axis']",
        r"$T_\mathrm{Sample}^\rightarrow$ (K)",
    ],
    "T_axis_down_K": [
        "self.mapped_over_temperature['temperature_axis']",
        r"$T_\mathrm{Sample}^\leftarrow$ (K)",
    ],
    "uH_up_mT": ["self.mapped['y_axis']*1e3", r"$\mu_0H^\rightarrow$ (mT)"],
    "uH_up_T": ["self.mapped['y_axis']", r"$\mu_0H^\rightarrow$ (T)"],
    "uH_down_mT": ["self.mapped['y_axis']*1e3", r"$\mu_0H^\leftarrow$ (mT)"],
    "uH_down_T": ["self.mapped['y_axis']", r"$\mu_0H^\leftarrow$ (T)"],
    "uH_mT": ["self.mapped['y_axis']*1e3", r"$\mu_0H$ (mT)"],
    "uH_T": ["self.mapped['y_axis']", r"$\mu_0H$ (T)"],
    "V_gate_up_V": ["self.mapped['y_axis']", r"$V_\mathrm{Gate}^\rightarrow$ (V)"],
    "V_gate_down_V": ["self.mapped['y_axis']", r"$V_\mathrm{Gate}^\leftarrow$ (V)"],
    "V_gate_V": ["self.mapped['y_axis']", r"$V_\mathrm{Gate}$ (V)"],
    "V_gate_up_mV": [
        "self.mapped['y_axis']*1e3",
        r"$V_\mathrm{Gate}^\rightarrow$ (mV)",
    ],
    "V_gate_down_mV": [
        "self.mapped['y_axis']*1e3",
        r"$V_\mathrm{Gate}^\leftarrow$ (mV)",
    ],
    "V_gate_mV": ["self.mapped['y_axis']*1e3", r"$V_\mathrm{Gate}$ (mV)"],
    "time_up": ["self.mapped['time_up']", r"time"],
    "V_ac_up_V": ["self.mapped['y_axis']", r"$V_\mathrm{AC}^\rightarrow$ (V)"],
    "nu_up": ["self.mapped['y_axis']*1e-9", r"$\nu_\mathrm{AC}^\rightarrow$ (GHz)"],
    "T_all_up_K": ["self.mapped['temperature_all_up']", r"$T_\mathrm{Sample}$ (K)"],
    "T_all_norm_up": [
        "self.mapped['temperature_all_up']/self.mapped['temperature_mean_up'].reshape(-1,1)",
        r"$T_\mathrm{Sample}(V_\mathrm{Bias}, V_\mathrm{AC})/T_\mathrm{Sample}(V_\mathrm{AC})$",
    ],
}


def plot_test_map():
    """Docstring"""
    match platform.system():
        case "Darwin":
            file_directory = "/User/oliver/Documents/p5control-bluefors-evaluation/utilities/blueforslogo.png"
        case "Linux":
            file_directory = "/home/oliver/Documents/p5control-bluefors-evaluation/utilities/blueforslogo.png"
        case default:
            logger.warning("(baseplotting) no system %s", default)
    try:

        image = Image.open(
            file_directory,
            mode="r",
        )
    except FileNotFoundError:
        logger.warning("(baseplotting) showTestMap() Trick verreckt :/")
        return
    image = np.asarray(image, dtype="float64")
    z_data = np.flip(image[:, :, 1], axis=0)
    z_data[z_data >= 80] = 0.8
    z_data /= np.max(z_data)

    fig, _, _ = plot_map(
        x=np.arange(image.shape[1]),
        y=np.arange(image.shape[0]),
        z=z_data,
        x_lim=[0.0, 2000.0],
        y_lim=[0.0, 1000.0],
        z_lim=[0.0, 1.0],
        x_label=r"$x_\mathrm{}$ (pxl)",
        y_label=r"$y_\mathrm{}$ (pxl)",
        z_label=r"BlueFors (arb. u.)",
        fig_nr=100,
        color_map=cmap(color="seeblau", bad="gray"),
        display_dpi=100,
        contrast=None,
    )
    fig.suptitle("Hier könnte ihre Werbung stehen.")


def get_ext(x, y, x_lim, y_lim):
    """Docstring"""

    pixel_width = np.abs(x[-1] - x[-2])
    pixel_height = np.abs(y[-1] - y[-2])
    ext = np.array(
        [
            x[0] - pixel_width / 2,
            x[len(x) - 1] + pixel_width / 2,
            y[0] - pixel_height / 2,
            y[len(y) - 1] + pixel_height / 2,
        ],
        dtype="float64",
    )

    new_x_lim = [None, None]
    new_y_lim = [None, None]
    if x_lim[0] is not None:
        new_x_lim[0] = x_lim[0] - pixel_width / 2
    if x_lim[1] is not None:
        new_x_lim[1] = x_lim[1] + pixel_width / 2
    if y_lim[0] is not None:
        new_y_lim[0] = y_lim[0] - pixel_height / 2
    if y_lim[1] is not None:
        new_y_lim[1] = y_lim[1] + pixel_height / 2

    return ext, new_x_lim, new_y_lim


def plot_map(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_lim: list,
    y_lim: list,
    z_lim: list,
    x_label: str = r"$x$-label",
    y_label: str = r"$y$-label",
    z_label: str = r"$z$-label",
    inverted: bool = False,
    fig_nr: int = 0,
    color_map=cmap(color="seeblau", bad="gray"),
    display_dpi: int = 100,
    contrast=None,
):
    """
    Docstring
    """

    if z.dtype == np.dtype("int32"):
        logger.warning("z is integer. Sure?")

    if contrast is not None:
        z_lim = [
            float(np.nanmean(z) - np.nanstd(z) / contrast),
            float(np.nanmean(z) + np.nanstd(z) / contrast),
        ]

    plt.close(fig_nr)
    fig, (ax_z, ax_c) = plt.subplots(
        num=fig_nr,
        ncols=2,
        figsize=(6, 4),
        dpi=display_dpi,
        gridspec_kw={"width_ratios": [5.8, 0.2]},
        constrained_layout=True,
    )

    if not inverted:
        ext, x_lim, y_lim = get_ext(x=x, y=y, x_lim=x_lim, y_lim=y_lim)

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

    else:
        ext, x_lim, y_lim = get_ext(x=y, y=x, x_lim=y_lim, y_lim=x_lim)

        im = ax_z.imshow(
            z.T,
            extent=ext,
            aspect="auto",
            origin="lower",
            clim=z_lim,
            cmap=color_map,
            interpolation="none",
        )
        ax_z.set_xlabel(y_label)
        ax_z.set_ylabel(x_label)

    ax_z.ticklabel_format(axis="both", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_z.tick_params(direction="in", right=True, top=True)
    ax_c.tick_params(direction="in", left=True)

    _ = ax_z.set_xlim(x_lim)
    _ = ax_z.set_ylim(y_lim)
    _ = fig.colorbar(im, label=z_label, cax=ax_c)

    return fig, ax_z, ax_c


def plot_map_vector(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n: np.ndarray,
    x_lim: list,
    y_lim: list,
    z_lim: list,
    n_lim: list,
    x_label: str = r"$x$-label",
    y_label: str = r"$y$-label",
    z_label: str = r"$z$-label",
    n_label: str = r"$n$-label",
    inverted: bool = False,
    fig_nr: int = 1,
    color_map=cmap(color="seeblau", bad="gray"),
    display_dpi: int = 100,
    contrast=None,
    vector_color=None,
    vector_style="-",
    vector_lwms=1,
):
    """
    Docstring
    """

    if z.dtype == np.dtype("int32"):
        logger.warning("z is integer. Sure?")

    if contrast is not None:
        z_lim = [
            float(np.nanmean(z) - np.nanstd(z) / contrast),
            float(np.nanmean(z) + np.nanstd(z) / contrast),
        ]

    plt.close(fig_nr)

    if not inverted:
        fig, (ax_n, ax_z, ax_c) = plt.subplots(
            num=fig_nr,
            ncols=3,
            figsize=(6, 4),
            dpi=display_dpi,
            gridspec_kw={"width_ratios": [1, 4.8, 0.2]},
            constrained_layout=True,
        )

        ext, x_lim, y_lim = get_ext(x=x, y=y, x_lim=x_lim, y_lim=y_lim)

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
        ax_z.ticklabel_format(
            axis="x", style="sci", scilimits=(-9, 9), useMathText=True
        )
        ax_z.tick_params(direction="in", right=True, top=True)
        ax_z.set_yticklabels([])

        ax_n.plot(
            n, y, vector_style, color=vector_color, ms=vector_lwms, lw=vector_lwms
        )
        ax_n.set_ylabel(y_label)
        ax_n.ticklabel_format(
            axis="x", style="sci", scilimits=(-9, 9), useMathText=True
        )
        ax_n.tick_params(direction="in", right=True, top=True)
        ax_n.grid()
        ax_n.set_xlabel(n_label)
        ax_n.invert_xaxis()
        ax_c.tick_params(direction="in")

        _ = fig.colorbar(im, label=z_label, cax=ax_c)
        _ = ax_z.set_xlim(x_lim)
        _ = ax_z.set_ylim(y_lim)

        y_lim = ax_z.get_ylim()
        _ = ax_n.set_xlim(n_lim)
        _ = ax_n.set_ylim(y_lim)

        ax_z.sharey(ax_n)
        plt.setp(ax_z.get_yticklabels(), visible=False)

    else:
        fig, axs = plt.subplots(
            num=fig_nr,
            ncols=2,
            nrows=2,
            figsize=(6, 4),
            dpi=display_dpi,
            gridspec_kw={"width_ratios": [4.8, 0.2], "height_ratios": [4, 1]},
            constrained_layout=True,
        )
        ax_z = axs[0, 0]
        ax_n = axs[1, 0]
        gs = axs[0, 1].get_gridspec()
        axs[0, 1].remove()
        axs[1, 1].remove()
        ax_c = fig.add_subplot(gs[0:, -1])

        ext, x_lim, y_lim = get_ext(x=y, y=x, x_lim=y_lim, y_lim=x_lim)

        im = ax_z.imshow(
            z.T,
            extent=ext,
            aspect="auto",
            origin="lower",
            clim=z_lim,
            cmap=color_map,
            interpolation="none",
        )
        ax_z.set_ylabel(x_label)
        ax_z.ticklabel_format(
            axis="y", style="sci", scilimits=(-9, 9), useMathText=True
        )
        ax_z.tick_params(direction="in", right=True, top=True)
        ax_z.set_xticklabels([])

        ax_n.plot(
            y, n, vector_style, color=vector_color, ms=vector_lwms, lw=vector_lwms
        )
        ax_n.set_xlabel(y_label)
        ax_n.set_ylabel(n_label)
        ax_n.ticklabel_format(
            axis="y", style="sci", scilimits=(-9, 9), useMathText=True
        )
        ax_n.tick_params(direction="in", right=True, top=True)
        ax_n.grid()
        ax_c.tick_params(direction="in")

        _ = fig.colorbar(im, label=z_label, cax=ax_c)
        _ = ax_z.set_xlim(x_lim)
        _ = ax_z.set_ylim(y_lim)

        x_lim = ax_z.get_xlim()
        _ = ax_n.set_xlim(x_lim)
        _ = ax_n.set_ylim(n_lim)

        ax_z.sharex(ax_n)
        plt.setp(ax_z.get_xticklabels(), visible=False)

    return fig, ax_z, ax_c, ax_n
