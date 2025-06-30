# region imports
# std
from typing import Any
from enum import Enum, auto

# third-party
from matplotlib import pyplot as plt
import numpy as np

# local
import utilities.logging as Logger
from integration.files import DataCollection
from .corporate_design_colors_v4 import cmap

# endregion


# region constants
class StyleKeys(Enum):
    X_LABEL = auto()
    Y_LABEL = auto()
    Z_LABEL = auto()

    SCALE = auto()
    X_LIM = auto()
    Y_LIM = auto()
    Z_LIM = auto()

    X_TICKS = auto()
    Y_TICKS = auto()
    Z_TICKS = auto()

    CMAP = auto()
    CBAR = auto()
    INTERPOL = auto()

    ASPECT = auto()

    FLIP_VERTICAL = auto()
    FLIP_HORIZONTAL = auto()


X_LABEL = StyleKeys.X_LABEL
Y_LABEL = StyleKeys.Y_LABEL
Z_LABEL = StyleKeys.Z_LABEL
SCALE = StyleKeys.SCALE
X_LIM = StyleKeys.X_LIM
Y_LIM = StyleKeys.Y_LIM
Z_LIM = StyleKeys.Z_LIM
X_TICKS = StyleKeys.X_TICKS
Y_TICKS = StyleKeys.Y_TICKS
Z_TICKS = StyleKeys.Z_TICKS
CMAP = StyleKeys.CMAP
CBAR = StyleKeys.CBAR
INTERPOL = StyleKeys.INTERPOL
ASPECT = StyleKeys.ASPECT
FLIP_VERTICAL = StyleKeys.FLIP_VERTICAL
FLIP_HORIZONTAL = StyleKeys.FLIP_HORIZONTAL
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


def map(bib: DataCollection, type: list[str], styling: list[dict[StyleKeys, Any]]):
    if len(type) != len(styling):
        Logger.print(
            Logger.ERROR,
            msg=f"Type and styling must be the same length ({len(type)} != {len(styling)})",
        )
        return

    for i, t in enumerate(type):
        match t:
            case "VXI" | "IXV" | "dIXR" | "dVXC":
                Logger.print(
                    Logger.DEBUG,
                    msg=f"Plotting map: {t}",
                )

                style = styling[i]
                map_def = bib.result.maps[t]
                xcoords_def = map_def.x_axis.values
                ycoords_def = map_def.y_axis.values

                xcoords = xcoords_def.copy()
                ycoords = ycoords_def.copy()
                map = map_def.copy()

                scale = style.get(SCALE)
                if scale is not None:
                    xcoords = xcoords_def * scale[0]
                    ycoords = ycoords_def * scale[1]
                    map.values = map_def.values * scale[2]

                if style.get(FLIP_VERTICAL, True):
                    map.values = np.flipud(map.values)

                if style.get(FLIP_HORIZONTAL, True):
                    map.values = np.fliplr(map.values)

                scmap = style.get(CMAP, "viridis")
                if scmap in ["seeblau"]:
                    scmap = cmap(clim=(-0.1, 1.0))

                print(map.values.shape)

                plt.imshow(
                    map.values,
                    cmap=scmap,
                    interpolation=style.get(INTERPOL, None),
                    aspect=style.get(ASPECT, "auto"),
                    extent=(
                        xcoords[0],
                        xcoords[-1],
                        ycoords[0],
                        ycoords[-1],
                    ),
                    clim=(2, 2.7),
                )

                style_map(map, style)

    plt.show()


def style_map(map, style: dict[StyleKeys, Any]):
    # region labels
    plt.xlabel(style.get(X_LABEL) or map.x_axis.name, fontsize=14)
    plt.ylabel(style.get(Y_LABEL) or map.y_axis.name, fontsize=14)
    if style.get(CBAR, False):
        plt.colorbar(label=style.get(Z_LABEL, map.z_axis.name))
    # endregion
    # region lims
    if style.get(X_LIM):
        plt.xlim(style.get(X_LIM))
    if style.get(Y_LIM):
        plt.ylim(style.get(Y_LIM))
    if style.get(Z_LIM):
        plt.clim(style.get(Z_LIM))
    # endregion
    # region ticks
    x_tick = style.get(X_TICKS)
    if x_tick:
        plt.xticks(np.arange(x_tick[0], x_tick[1] + x_tick[2], x_tick[2]))

    y_tick = style.get(Y_TICKS)
    if y_tick:
        plt.yticks(np.arange(y_tick[0], y_tick[1] + y_tick[2], y_tick[2]))
    plt.tick_params(axis="both", which="major", labelsize=14)
    # endregion
