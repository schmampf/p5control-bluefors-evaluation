"""
Description.

"""

import logging

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utilities.corporate_design_colors_v4 import cmap
from utilities.basefunctions import get_ext
from utilities.corporate_design_colors_v4 import COLORS

logger = logging.getLogger(__name__)

plt.ioff()


def plot_all(
    indices: list,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    i: np.ndarray,
    didv: np.ndarray,
    n: np.ndarray,
    x_lim: tuple,
    y_lim: tuple,
    z_lim: tuple,
    v_lim: float,
    i_lim: tuple,
    didv_lim: tuple,
    n_lim: tuple,
    x_label: str = r"$x$-label",
    y_label: str = r"$y$-label",
    z_label: str = r"$z$-label",
    v_label: str = r"$v$-label",
    i_label: str = r"$i$-label",
    didv_label: str = r"$didv$-label",
    n_label: str = r"$n$-label",
    fig_nr: int = 0,
    fig_size: tuple = (6, 4),
    display_dpi: int = 100,
    z_cmap=cmap(color="seeblau", bad="gray"),
    z_contrast=0.05,
    n_show=True,
    n_size=(0.3, 0.3),
    n_labelsize=None,
    n_color="grey",
    n_style="-",
    n_lwms=1,
    title=None,
):
    """
    Docstring
    """

    if z.dtype == np.dtype("int32"):
        logger.warning("z is integer. Sure?")

    if z_contrast is not None:
        delta_z = z_lim[1] - z_lim[0]
        z_lim = (z_lim[0] - z_contrast * delta_z, z_lim[1] + z_contrast * delta_z)

    z_ext, zx_lim, zy_lim = get_ext(x=x, y=y, x_lim=x_lim, y_lim=y_lim)

    if y_lim[0] is not None:
        y_indices_0 = np.argmin(np.abs(y - y_lim[0]))
    else:
        y_indices_0 = 0

    if y_lim[1] is not None:
        y_indices_1 = np.argmin(np.abs(y - y_lim[1]))
    else:
        y_indices_1 = -1

    if v_lim is None:
        upper_v_lim = np.argmin(np.abs(v - np.nanmax(v)))
        lower_v_lim = np.argmin(np.abs(-v - np.nanmax(-v)))
    else:
        upper_v_lim = np.argmin(np.abs(v - v_lim))
        lower_v_lim = np.argmin(np.abs(-v - v_lim))
    zero_v_lim = np.argmin(np.abs(v))

    # Generate Figure
    plt.close(fig_nr)
    fig, axs = plt.subplots(
        num=fig_nr,
        nrows=3,
        ncols=2,
        figsize=fig_size,
        dpi=display_dpi,
        gridspec_kw={"height_ratios": [0.2, 2.0, 1.2], "width_ratios": [4, 4]},
        constrained_layout=True,
    )

    for j, index in enumerate(indices):
        axs[1, 0].plot(v[zero_v_lim:upper_v_lim], i[index, zero_v_lim:upper_v_lim])
        axs[1, 0].plot(-v[lower_v_lim:zero_v_lim], -i[index, lower_v_lim:zero_v_lim])
        axs[2, 0].plot(v[zero_v_lim:upper_v_lim], didv[index, zero_v_lim:upper_v_lim])
        axs[2, 0].plot(-v[lower_v_lim:zero_v_lim], didv[index, lower_v_lim:zero_v_lim])
    temp_v_lim = axs[1, 0].get_xlim()
    temp_i_lim = list(axs[1, 0].get_ylim())
    temp_didv_lim = list(axs[2, 0].get_ylim())

    for i, lim in enumerate(i_lim):
        if lim is None:
            temp_i_lim[i] = lim

    for i, lim in enumerate(didv_lim):
        if lim is None:
            temp_didv_lim[i] = lim

    if n_show:
        axs[0, 1].plot(y[y_indices_0:y_indices_1], n[y_indices_0:y_indices_1])
        temp_n_lim = axs[0, 1].get_ylim()
        temp_y_lim = axs[0, 1].get_xlim()
        if n_lim[0] is not None:
            temp_n_lim_0 = n_lim[0]
        else:
            temp_n_lim_0 = temp_n_lim[0]
        if n_lim[1] is not None:
            temp_n_lim_1 = n_lim[1]
        else:
            temp_n_lim_1 = temp_n_lim[1]
        if y_lim[0] is not None:
            temp_y_lim_0 = y_lim[0]
        else:
            temp_y_lim_0 = temp_y_lim[0]
        if y_lim[1] is not None:
            temp_y_lim_1 = y_lim[1]
        else:
            temp_y_lim_1 = temp_y_lim[1]
        temp_n_lim = (temp_n_lim_0, temp_n_lim_1)
        temp_y_lim = (temp_y_lim_0, temp_y_lim_1)

    # Generate proper axs
    ax_didv = axs[2, 1]
    ax_c = axs[0, 0]

    gs_i = axs[0, 1].get_gridspec()
    axs[0, 1].remove()
    axs[1, 1].remove()
    ax_i = fig.add_subplot(gs_i[:2, 1])

    gs_z = axs[1, 0].get_gridspec()
    axs[1, 0].remove()
    axs[2, 0].remove()
    ax_z = fig.add_subplot(gs_z[1:, 0])

    if n_show:
        ax_n = inset_axes(
            ax_i,
            width=f"{int(n_size[0]*100):02d}%",  # width = 30% of parent_bbox
            height=f"{int(n_size[1]*100):02d}%",  # height : 1 inch
            loc="upper left",
        )

    # set tick style
    ax_z.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_z.tick_params(direction="in", right=True, top=True)

    ax_c.tick_params(direction="in")

    ax_i.ticklabel_format(axis="both", style="sci", scilimits=(-9, 9), useMathText=True)
    ax_i.tick_params(direction="in", right=True, top=True)
    # ax_i.set_xticklabels([])
    ax_i.xaxis.set_label_position("top")
    ax_i.yaxis.set_label_position("right")
    ax_i.yaxis.tick_right()
    ax_i.xaxis.tick_top()

    ax_didv.ticklabel_format(
        axis="both", style="sci", scilimits=(-9, 9), useMathText=True
    )
    ax_didv.tick_params(direction="in", right=True, top=True)
    ax_didv.yaxis.set_label_position("right")
    ax_didv.yaxis.tick_right()

    if n_show:
        ax_n.ticklabel_format(
            axis="both", style="sci", scilimits=(-9, 9), useMathText=True
        )
        ax_n.tick_params(direction="in", right=True, top=True, labelsize=n_labelsize)
        ax_n.yaxis.set_label_position("right")
        ax_n.xaxis.set_label_position("bottom")
        ax_n.yaxis.tick_right()

    # set label and grid
    ax_z.set_xlabel(x_label)
    ax_z.set_ylabel(y_label)
    ax_i.set_ylabel(i_label)
    ax_i.set_xlabel(v_label)
    ax_didv.set_xlabel(v_label)
    ax_didv.set_ylabel(didv_label)

    ax_i.grid()
    ax_didv.grid()

    if n_show:
        if n_labelsize is not None:
            ax_n.set_xlabel(y_label, fontsize=n_labelsize)
            ax_n.set_ylabel(n_label, fontsize=n_labelsize)
        else:
            ax_n.set_xlabel(y_label)
            ax_n.set_ylabel(n_label)
        ax_n.grid()

    # img and clb
    im = ax_z.imshow(
        z,
        extent=z_ext,
        aspect="auto",
        origin="lower",
        clim=z_lim,
        cmap=z_cmap,
        interpolation="none",
    )

    cb = fig.colorbar(im, label=z_label, cax=ax_c, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position("top")

    for j, index in enumerate(indices):

        ax_i.plot(
            v,
            i[index, :],
            "-",
            color=COLORS[j + 1],
            lw=n_lwms,
            ms=n_lwms,
        )
        ax_i.plot(
            -v,
            -i[index, :],
            "--",
            color=COLORS[j + 1],
            lw=n_lwms,
            ms=n_lwms,
        )
        ax_didv.plot(
            v,
            didv[index, :],
            "-",
            color=COLORS[j + 1],
            lw=n_lwms,
            ms=n_lwms,
        )
        ax_didv.plot(
            -v,
            didv[index, :],
            "--",
            color=COLORS[j + 1],
            lw=n_lwms,
            ms=n_lwms,
        )
        ax_z.plot(np.array(x_lim), (y[index], y[index]), lw=2, color=COLORS[j + 1])
    ax_i.plot(
        [1000, 1001],
        [1000, 1001],
        "k-",
        label="pos.",
    )
    ax_i.plot(
        [1000, 1001],
        [1000, 1001],
        "k--",
        label="neg.",
    )
    if n_labelsize is not None:
        ax_i.legend(loc="lower right", prop={"size": n_labelsize})
    else:
        ax_i.legend(loc="lower right")
    # set limits
    _ = ax_i.set_xlim(temp_v_lim)
    _ = ax_i.set_ylim(temp_i_lim)
    _ = ax_didv.set_xlim(temp_v_lim)
    _ = ax_didv.set_ylim(temp_didv_lim)

    _ = ax_z.set_xlim(left=zx_lim[0], right=zx_lim[1])
    _ = ax_z.set_ylim(bottom=zy_lim[0], top=zy_lim[1])

    if n_show:
        for j, index in enumerate(indices):
            ax_n.plot([y[index], y[index]], temp_n_lim, lw=2, color=COLORS[j])
        ax_n.plot(y, n, n_style, color=n_color, lw=n_lwms, ms=n_lwms)
        _ = ax_n.set_xlim(left=temp_y_lim[0], right=temp_y_lim[1])
        _ = ax_n.set_ylim(bottom=temp_n_lim[0], top=temp_n_lim[1])

        ax_n.yaxis.set_major_locator(
            MaxNLocator(nbins=3, min_n_ticks=2, steps=[1, 2, 2.5, 5, 10])
        )
        ax_n.xaxis.set_major_locator(
            MaxNLocator(nbins=3, min_n_ticks=2, steps=[1, 2, 2.5, 5, 10])
        )
        n_xticklabels = ax_n.get_xticklabels()
        n_xticks = ax_n.get_xticks()
        ax_n.set_xticks(n_xticks)
        n_yticks = ax_n.get_yticks()
        ax_n.set_yticks(n_yticks)
        n_yticklabels = ax_n.get_yticklabels()
        if n_labelsize is not None:
            ax_n.set_xticklabels(n_xticklabels, fontsize=n_labelsize)
            ax_n.set_yticklabels(n_yticklabels, fontsize=n_labelsize)
        else:
            ax_n.set_xticklabels(n_xticklabels)
            ax_n.set_yticklabels(n_yticklabels)
        _ = ax_n.set_xlim(left=temp_y_lim[0], right=temp_y_lim[1])
        _ = ax_n.set_ylim(bottom=temp_n_lim[0], top=temp_n_lim[1])
    else:
        ax_n = None

    ax_i.sharex(ax_didv)
    # plt.setp(ax_i.get_xticklabels(), visible=False)

    if title is not None:
        plt.suptitle(title)

    # plt.show()

    return fig, ax_z, ax_c, ax_n, ax_i, ax_didv
