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


def plot_map(
    bib: DataCollection,
    yz: tuple[str, str],
):
    num_slices = bib.params.available_measurement_entries
    x = bib.params.selected_measurement.variable.name

    ypath, y = yz[0].split("/")
    zpath, z = yz[1].split("/")

    init: bool = False
    map = np.empty((num_slices - 1, 0), dtype=float)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    z_vals = np.arange(num_slices - 1)

    Logger.suppress_print = True
    for i in range(1, num_slices):
        print(f"Processing {i} of {num_slices}")

        Macros.macro_eval(bib, i, ypath)
        resulty = bib.evaluation.cached_sets[ypath].curves[y]
        resultz = bib.evaluation.cached_sets[zpath].curves[z]

        if not init:
            map = np.empty((num_slices - 1, resulty.size), dtype=float)
            init = True

        map[i - 1, :] = bib.evaluation.cached_sets[ypath].curves[y]
        x = (np.full_like(resulty, z_vals[i - 1]),)
        ax.plot(x, resulty, resultz)

    Logger.suppress_print = False

    # plt.imshow(
    #     map,
    # )
    plt.gca().set_aspect("auto")
    plt.show()
