"""
Description.

"""

import logging
import platform

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from matplotlib.ticker import MaxNLocator
from utilities.corporate_design_colors_v4 import cmap

from utilities.basefunctions import get_ext

plt.ioff()

logger = logging.getLogger(__name__)


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

    plt.close(fig_nr + int(inverted))
    fig, (ax_z, ax_c) = plt.subplots(
        num=fig_nr + int(inverted),
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

    if not inverted:
        plt.close(fig_nr)
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
        _ = ax_n.set_xlim([n_lim[1], n_lim[0]])
        _ = ax_n.set_ylim(y_lim)

        ax_z.sharey(ax_n)
        plt.setp(ax_z.get_yticklabels(), visible=False)

    else:
        plt.close(fig_nr + 1)
        fig, axs = plt.subplots(
            num=fig_nr + 1,
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


def plot_iv(
    indices: list,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    n: np.ndarray,
    i: np.ndarray,
    x_lim: list,
    y_lim: list,
    z_lim: list,
    n_lim: list,
    i_lim: float,
    x_label: str = r"$x$-label",
    y_label: str = r"$y$-label",
    z_label: str = r"$z$-label",
    n_label: str = r"$n$-label",
    i_label: str = r"$i$-label",
    fig_nr: int = 2,
    display_dpi: int = 100,
    vector_color="grey",
    vector_style="-",
    vector_lwms=1,
):
    """
    Docstring
    """

    plt.close(fig_nr)
    fig, axs = plt.subplots(
        num=fig_nr,
        nrows=2,
        ncols=2,
        figsize=(6, 4),
        dpi=display_dpi,
        gridspec_kw={"height_ratios": [3, 2], "width_ratios": [5, 1]},
        constrained_layout=True,
    )

    ax_i = axs[0, 0]
    ax_didv = axs[1, 0]
    gs = axs[0, 1].get_gridspec()

    axs[0, 1].plot(n, y)
    n_lim = axs[0, 1].get_xlim()
    axs[0, 1].remove()
    axs[1, 1].remove()
    ax_y = fig.add_subplot(gs[0:, -1])

    ax_i.ticklabel_format(axis="both", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_i.tick_params(direction="in", right=True, top=True)
    ax_i.set_xticklabels([])
    ax_didv.ticklabel_format(
        axis="both", style="sci", scilimits=(-9, 9), useMathText=True
    )
    ax_didv.tick_params(direction="in", right=True, top=True)
    ax_y.ticklabel_format(axis="both", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_y.tick_params(direction="in", right=True, top=True)
    ax_y.yaxis.set_label_position("right")
    ax_y.yaxis.tick_right()
    ax_y.invert_xaxis()
    ax_y.xaxis.set_major_locator(MaxNLocator(2))
    ax_y.set_yticks(y[indices])

    ax_didv.set_xlabel(x_label)
    ax_didv.set_ylabel(z_label)
    ax_i.set_ylabel(i_label)
    ax_y.set_xlabel(n_label)
    ax_y.set_ylabel(y_label)

    ax_i.grid()
    ax_didv.grid()
    ax_y.grid()

    for j, index in enumerate(indices):
        ax_i.plot(
            x,
            i[index, :],
            "-",
            label=f"{y_label} = {y[index]:04.02}",  #
            color=COLORS[j],
        )
        ax_i.plot(
            -x,
            -i[index, :],
            "--",
            label=f"{y_label} = {y[index]:04.02}",  #
            color=COLORS[j],
        )
        ax_didv.plot(
            x, z[index, :], "-", label=f"{n_label} = {n[index]:04.02}", color=COLORS[j]
        )
        ax_didv.plot(
            -x,
            z[index, :],
            "--",
            label=f"{n_label} = {n[index]:04.02}",
            color=COLORS[j],
        )
        ax_y.plot(n_lim, [y[index], y[index]], lw=2, color=COLORS[j])
    ax_y.plot(n, y, vector_style, color=vector_color, lw=vector_lwms, ms=vector_lwms)

    _ = ax_i.set_xlim([0, x_lim[1]])
    _ = ax_i.set_ylim([0, i_lim])
    _ = ax_didv.set_xlim([0, x_lim[1]])
    _ = ax_didv.set_ylim(z_lim)
    _ = ax_y.set_xlim((float(n_lim[1]), float(n_lim[0])))
    _ = ax_y.set_ylim(None)

    ax_i.sharex(ax_didv)
    plt.setp(ax_i.get_xticklabels(), visible=False)

    return fig, ax_i, ax_didv, ax_y
