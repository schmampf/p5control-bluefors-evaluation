"""
Pre-Definition of Functions
"""

import numpy as np
from torch.nn import Upsample
from torch import from_numpy

import matplotlib.pyplot as plt

import warnings

from ..corporate_design_colors_v4 import cmap

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


def linfit(x):
    # time binned over voltage is not equally spaced and might have nans
    # this function does a linear fit to uneven spaced array, that might contain nans
    # gives back linear and even spaced array
    nu_x = np.copy(x)
    nans = np.isnan(x)
    not_nans = np.invert(nans)
    xx = np.arange(np.shape(nu_x)[0])
    poly = np.polyfit(xx[not_nans], nu_x[not_nans], 1)
    fit_x = xx * poly[0] + poly[1]
    return fit_x


def bin_y_over_x(
    x,
    y,
    x_bins,
    upsampling=None,
):
    # gives y-values over even-spaced and monoton-increasing x
    # incase of big gaps in y-data, use upsampling, to fill those.
    if upsampling is not None:
        k = np.full((2, len(x)), np.nan)
        k[0, :] = x
        k[1, :] = y
        m = Upsample(mode="linear", scale_factor=upsampling)
        big = m(from_numpy(np.array([k])))
        x = np.array(big[0, 0, :])
        y = np.array(big[0, 1, :])
    else:
        pass

    # Apply binning based on histogram function
    x_nu = np.append(x_bins, 2 * x_bins[-1] - x_bins[-2])
    x_nu = x_nu - (x_nu[1] - x_nu[0]) / 2
    # Instead of N_x, gives fixed axis.
    # Solves issues with wider ranges, than covered by data
    _count, _ = np.histogram(x, bins=x_nu, weights=None)
    _count = np.array(_count, dtype="float64")
    _count[_count == 0] = np.nan

    _sum, _ = np.histogram(x, bins=x_nu, weights=y)
    return _sum / _count, _count


def bin_z_over_y(
    y,
    z,
    y_binned,
):
    N_bins = np.shape(y_binned)[0]
    counter = np.full(N_bins, 0)
    result = np.full((N_bins, np.shape(z)[1]), 0, dtype="float64")

    # Find Indizes of x on x_binned
    dig = np.digitize(y, bins=y_binned)

    # Add up counter, I & differential_conductance
    for i, d in enumerate(dig):
        counter[d - 1] += 1
        result[d - 1, :] += z[i, :]

    # Normalize with counter, rest to np.nan
    for i, c in enumerate(counter):
        if c > 0:
            result[i, :] /= c
        elif c == 0:
            result[i, :] *= np.nan

    # Fill up Empty lines with Neighboring Lines
    for i, c in enumerate(counter):
        if c == 0:  # In case counter is 0, we need to fill up
            up, down = i, i  # initialze up, down
            while counter[up] == 0 and up < N_bins - 1:
                # while up is still smaller -2 and counter is still zero, look for better up
                up += 1
            while counter[down] == 0 and down >= 1:
                # while down is still bigger or equal 1 and coutner still zero, look for better down
                down -= 1

            if up == N_bins - 1 or down == 0:
                # Just ignores the edges, when c == 0
                result[i, :] *= np.nan
            else:
                # Get Coordinate System
                span = up - down
                relative_pos = i - down
                lower_span = span * 0.25
                upper_span = span * 0.75

                # Divide in same as next one and intermediate
                if 0 <= relative_pos <= lower_span:
                    result[i, :] = result[down, :]
                elif lower_span < relative_pos < upper_span:
                    result[i, :] = (result[up, :] + result[down, :]) / 2
                elif upper_span <= relative_pos <= span:
                    result[i, :] = result[up, :]
                else:
                    warnings.warn("something went wrong!")
    return result, counter


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
        warnings.warn("z is integer. Sure?")

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
        warnings.warn("First of xy_lim must be smaller than first one.")

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


def plot_map_and_vector(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n: np.ndarray,
    x_lim: list[float] = [-1.0, 1.0],
    y_lim: list[float] = [-1.0, 1.0],
    z_lim: list[float] = [0.0, 0.0],
    n_lim: list[float] = [-1.0, 1.0],
    x_label: str = r"$x$-label",
    y_label: str = r"$y$-label",
    z_label: str = r"$z$-label",
    n_label: str = r"$n$-label",
    fig_nr: int = 0,
    cmap=cmap(color="seeblau", bad="gray"),
    display_dpi: int = 100,
    contrast: float = 1.0,
):

    if z.dtype == np.dtype("int32"):
        warnings.warn("z is integer. Sure?")

    if x_lim[0] >= x_lim[1] or y_lim[0] >= y_lim[1] or z_lim[0] >= z_lim[1]:
        warnings.warn("First of xy_lim must be smaller than first one.")

    if z_lim == [0, 0]:
        z_lim = [
            float(np.nanmean(z) - np.nanstd(z) / contrast),
            float(np.nanmean(z) + np.nanstd(z) / contrast),
        ]

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
    n = np.array(n[y_ind[0] : y_ind[1]], dtype="float64")

    plt.close(fig_nr)
    fig, (ax_n, ax_z, ax_c) = plt.subplots(
        num=fig_nr,
        ncols=3,
        figsize=(6, 4),
        dpi=display_dpi,
        gridspec_kw={"width_ratios": [1, 4.8, 0.2]},
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
    ax_z.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_z.tick_params(direction="in", right=True, top=True)
    ax_z.set_yticklabels([])

    ax_n.plot(n, y)
    ax_n.set_ylabel(y_label)
    ax_n.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_n.tick_params(direction="in", right=True, top=True)
    ax_n.grid()
    ax_n.set_xlabel(n_label)

    cbar = fig.colorbar(im, label=z_label, cax=ax_c)
    ax_c.tick_params(direction="in")
    lim = ax_z.set_xlim(ext[0], ext[1])
    lim = ax_z.set_ylim(ext[2], ext[3])
    lim = ax_n.set_ylim(ext[2], ext[3])

    return fig, ax_z, ax_c, x, y, z, ext


def plot_map_and_vector_2(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n: np.ndarray,
    x_lim: list[float] = [-1.0, 1.0],
    y_lim: list[float] = [-1.0, 1.0],
    z_lim: list[float] = [0.0, 0.0],
    n_lim: list[float] = [-1.0, 1.0],
    x_label: str = r"$x$-label",
    y_label: str = r"$y$-label",
    z_label: str = r"$z$-label",
    n_label: str = "test",
    fig_nr: int = 0,
    cmap=cmap(color="seeblau", bad="gray"),
    display_dpi: int = 100,
    contrast: float = 1.0,
):

    if z.dtype == np.dtype("int32"):
        warnings.warn("z is integer. Sure?")

    if x_lim[0] >= x_lim[1] or y_lim[0] >= y_lim[1] or z_lim[0] >= z_lim[1]:
        warnings.warn("First of xy_lim must be smaller than first one.")

    if z_lim == [0, 0]:
        z_lim = [
            float(np.nanmean(z) - np.nanstd(z) / contrast),
            float(np.nanmean(z) + np.nanstd(z) / contrast),
        ]

    stepsize_x = np.abs(x[-1] - x[-2]) / 2
    stepsize_y = np.abs(y[-1] - y[-2]) / 2
    # stepsize_x=0#np.abs(x[-1]-x[-2])/2
    # stepsize_y=0#np.abs(y[-1]-y[-2])/2
    x_ind = [np.abs(x - x_lim[0]).argmin(), np.abs(x - x_lim[1]).argmin()]
    y_ind = [np.abs(y - y_lim[0]).argmin(), np.abs(y - y_lim[1]).argmin()]

    ext = np.array(
        [
            y[y_ind[0]] - stepsize_y,
            y[y_ind[1]] + stepsize_y,
            x[x_ind[0]] - stepsize_x,
            x[x_ind[1]] + stepsize_x,
        ],
        dtype="float64",
    )
    z = np.array(z[y_ind[0] : y_ind[1], x_ind[0] : x_ind[1]], dtype="float64")
    x = np.array(x[x_ind[0] : x_ind[1]], dtype="float64")
    y = np.array(y[y_ind[0] : y_ind[1]], dtype="float64")
    n = np.array(n[y_ind[0] : y_ind[1]], dtype="float64")

    plt.close(fig_nr)
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
    ax_c = axs[0, 1]
    axs[1, 1].remove()

    im = ax_z.imshow(
        np.rot90(z),
        extent=ext,
        aspect="auto",
        origin="lower",
        clim=z_lim,
        cmap=cmap,
        interpolation="none",
    )
    ax_z.set_ylabel(x_label)
    ax_z.ticklabel_format(axis="y", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_z.tick_params(direction="in", right=True, top=True)
    # ax_z.set_xticklabels([])

    ax_n.plot(y, n, ".")
    ax_n.set_ylabel(y_label)
    ax_n.ticklabel_format(axis="y", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_n.tick_params(direction="in", right=True, top=True)
    ax_n.grid()

    cbar = fig.colorbar(im, label=z_label, cax=ax_c)
    ax_c.tick_params(direction="in")
    # lim = ax_z.set_xlim(ext[0],ext[1])
    # lim = ax_z.set_ylim(ext[2],ext[3])

    # lim = ax_n.set_xlim(ext[0],ext[1])
    # lim = ax_n.set_ylim(n_lim)

    ax_n.sharex(ax_z)

    return fig, ax_z, ax_c, x, y, z, ext
