# region imports
# std

# third-party
from matplotlib import pyplot as plt
import numpy as np

# local
import evaluation.iv as IVEval
import evaluation.general as GenEval
import utilities.logging as Logger
import integration.files as Files
from integration.files import DataCollection
import algorithms.binning as Binning
import utilities.macros as Macros

# endregion


def plot_curves(
    bib: DataCollection,
    select: tuple[str, str],
    curves: list[tuple[str, ...]],
    show: bool = False,
):
    Logger.print(
        Logger.INFO,
        msg=f"Plot.plot_curves: {select} {curves}",
    )

    group = select[0]
    set = select[1]

    match group:
        case "raw":
            dataset = bib.evaluation.raw_sets[set]
        case "cache":
            dataset = bib.evaluation.cached_sets[set]
        case "filtered":
            dataset = bib.evaluation.filtered_sets[set]
        case "processed":
            dataset = bib.evaluation.processed_sets[set]

    for curve in curves:
        if len(curve) != 2:
            plt.plot(dataset.curves[curve[0]])
        elif len(curve) == 2:
            x = dataset.curves[curve[0]]
            y = dataset.curves[curve[1]]
            plt.plot(x, y, label=curve[1])
        else:
            raise ValueError("Curve must be a tuple of length 1 or 2")

    if show:
        plt.show()


# def plot_map(bib: DataCollection, xy: tuple[str, str], style: dict[str, str]):
#     """
#     Plot a map of the selected measurement.\n
#     Parameters
#     ----------
#     xy: paths to the x and y axes, each given by "dataset/curve", z axis given by measurement variable
#     """

#     plt.imshow(
#         map[:, :],
#         cmap="viridis",
#         interpolation="nearest",
#     )

#     x_tick_loc = np.linspace(0, map.shape[1] - 1, 11)
#     plt.xticks(
#         x_tick_loc,
#         [f"{x:.2f}" for x in x_values.flatten()[x_tick_loc.astype(int)]],
#     )
#     y_tick_loc = np.linspace(0, num_slices - 1, 11)
#     plt.yticks(
#         y_tick_loc,
#         [f"{z:.2f}" for z in z_values.flatten()[y_tick_loc.astype(int)]],
#     )
#     print(z_values.flatten())

#     if style:
#         unpack_style(style)
#     plt.gca().set_aspect("auto")
#     plt.show()


def get_unit(name: str) -> str:
    """
    Get the unit of a given name.
    """
    if name == "current":
        return "A"
    elif name == "voltage":
        return "V"
    elif name == "time":
        return "s"
    else:
        return ""


def unpack_style(
    style: dict[str, str],
):
    plt.xlabel(style.get("x-axis", "err"))
    plt.ylabel(style.get("y-axis", "err"))
    plt.colorbar(label=style.get("z-axis", "err"))
