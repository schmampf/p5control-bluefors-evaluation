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

# endregion


# region constants
class StyleKeys(Enum):
    X_LABEL = auto()
    Y_LABEL = auto()
    Z_LABEL = auto()

    X_LIM = auto()
    Y_LIM = auto()
    Z_LIM = auto()

    CMAP = auto()
    CBAR = auto()
    INTERPOL = auto()

    ASPECT = auto()
    TICKS = auto()


X_LABEL = StyleKeys.X_LABEL
Y_LABEL = StyleKeys.Y_LABEL
Z_LABEL = StyleKeys.Z_LABEL
X_LIM = StyleKeys.X_LIM
Y_LIM = StyleKeys.Y_LIM
Z_LIM = StyleKeys.Z_LIM
CMAP = StyleKeys.CMAP
CBAR = StyleKeys.CBAR
INTERPOL = StyleKeys.INTERPOL
ASPECT = StyleKeys.ASPECT
TICKS = StyleKeys.TICKS
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
            case "VVI":
                style = styling[i]
                map = bib.result.maps["VVI"]
                xcoords = map.x_axis.values
                ycoords = map.y_axis.values

                plt.imshow(
                    map.values,
                    cmap=style.get(CMAP, "viridis"),
                    interpolation=style.get(INTERPOL, None),
                    aspect=style.get(ASPECT, "auto"),
                    extent=(
                        xcoords[0],
                        xcoords[-1],
                        ycoords[0],
                        ycoords[-1],
                    ),
                )
                # region labels
                plt.xlabel(style.get(X_LABEL) or map.x_axis.name)
                plt.ylabel(style.get(Y_LABEL) or map.y_axis.name)
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
                ticks = style.get(TICKS)
                if ticks:
                    plt.locator_params(axis="x", nbins=ticks[0])
                    plt.locator_params(axis="y", nbins=ticks[1])
                    # if style.get(CBAR, False):
                    #     cbar = plt.colorbar(label=style.get(Z_LABEL, map.z_axis.name))
                    #     cbar_ticks = style.get("CBAR_TICKS")  # Add a key for color bar ticks
                    #     if cbar_ticks:
                    #         cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, cbar_ticks))

                # endregion

    plt.show()


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
