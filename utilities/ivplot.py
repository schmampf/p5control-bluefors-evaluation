# region imports
import sys
import logging
import importlib

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from tqdm import tqdm
from contextlib import contextmanager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from utilities.baseplot import BasePlot
from utilities.ivevaluation import IVEvaluation
from utilities.basefunctions import get_norm
from utilities.basefunctions import get_ext
from utilities.basefunctions import get_z_lim
from utilities.corporate_design_colors_v4 import cmap


importlib.reload(sys.modules["utilities.baseplot"])
importlib.reload(sys.modules["utilities.ivevaluation"])
importlib.reload(sys.modules["utilities.basefunctions"])
importlib.reload(sys.modules["utilities.corporate_design_colors_v4"])

logger = logging.getLogger(__name__)


from scipy.signal import savgol_filter
from scipy.interpolate import NearestNDInterpolator


# Increase the limit
# Or disable the warning entirely
mpl.rcParams["figure.max_open_warning"] = 0

# endregion


# region functions


@contextmanager
def suppress_logging():
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def do_smoothing(data: np.ndarray, window_length: int = 20, polyorder: int = 2):
    is_nan = np.isnan(data)
    mask = np.where(np.logical_not(is_nan))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    data = interp(*np.indices(data.shape))

    data = savgol_filter(
        data,
        window_length=window_length,
        polyorder=polyorder,
    )

    data[is_nan] = np.nan

    return data


# endregion


class IVPlot(IVEvaluation, BasePlot):

    def __init__(
        self,
        name: str = "iv plot",
    ):
        self._iv_plot_name = name

        # Initialize both parental classes
        IVEvaluation.__init__(self)
        BasePlot.__init__(self)

        self.to_plot = self.get_empty_dictionary()
        self.title_of_plot = ""
        self.y_characters = ["$y$", ""]

        self.iv_plot = {
            "fig_size": (6, 4),
            "dpi": 100,
            "cmap": cmap(bad="grey"),
            "smoothing": False,
            "window_length": 20,
            "polyorder": 2,
        }

        self.plot_ivs = False
        self.plot_didvs = True
        self.plot_dvdis = True
        self.plot_T = True
        self.plot_index = 0
        self.contrast = 1

        self.plot_down_sweep = False

        self.y_lim: tuple[float | None, float | None] = (None, None)
        self.i_lim: tuple[float | None, float | None] = (None, None)
        self.v_lim: tuple[float | None, float | None] = (None, None)
        self.t_lim: tuple[float | None, float | None] = (None, None)
        self.T_lim: tuple[float | None, float | None] = (None, None)
        self.didv_lim: tuple[float | None, float | None] = (None, None)
        self.dvdi_lim: tuple[float | None, float | None] = (None, None)
        self.didv_c_lim: tuple[float | None, float | None] = (None, None)
        self.dvdi_c_lim: tuple[float | None, float | None] = (None, None)

        self.y_norm: tuple[float, str] | None = None
        self.i_norm: tuple[float, str] | None = None
        self.v_norm: tuple[float, str] | None = None
        self.t_norm: tuple[float, str] | None = None
        self.T_norm: tuple[float, str] | None = None
        self.didv_norm: tuple[float, str] | None = None
        self.dvdi_norm: tuple[float, str] | None = None

        logger.info("(%s) ... IVPlot initialized.", self._iv_plot_name)

    # region get data

    def loadData(self):
        # Call the parent class's loadData method
        super().loadData()

        if not self.plot_down_sweep:
            self.to_plot = self.up_sweep
            self.title_of_plot = "Up Sweep"
        else:
            self.to_plot = self.down_sweep
            self.title_of_plot = "Down Sweep"

    def get_y_len(self):
        return self.mapped["y_axis"].shape[0]

    def get_y(
        self,
        index: int = 0,
        plain: bool = False,
    ):
        if plain:
            return np.nan
        else:
            return self.mapped["y_axis"][index]

    def get_T(
        self,
        index: int = 0,
        plain: bool = False,
    ):
        if plain:
            return self.to_plot["plain"]["temperature"][0]
        else:
            return self.to_plot["temperature"][index]

    def get_ivt(
        self,
        index: int = 0,
        plain: bool = False,
        skip: list[int] = [10, -10],
    ):
        if plain:
            ivt = self.to_plot["plain"]["iv_tuples"][0]
            i = np.copy(ivt[0][skip[0] : skip[1]])
            v = np.copy(ivt[1][skip[0] : skip[1]])
            t = np.copy(ivt[2][skip[0] : skip[1]])
        else:
            ivt = self.to_plot["iv_tuples"][index]
            i = np.copy(ivt[0])
            v = np.copy(ivt[1])
            t = np.copy(ivt[2])
        return i, v, t

    def get_ivt_raw(
        self,
        index: int = 0,
        plain: bool = False,
        skip: list[int] = [10, -10],
    ):
        if plain:
            ivt = self.to_plot["plain"]["iv_tuples_raw"][0]
            i = np.copy(ivt[0][skip[0] : skip[1]])
            v = np.copy(ivt[1][skip[0] : skip[1]])
            t = np.copy(ivt[2][skip[0] : skip[1]])
        else:
            ivt = self.to_plot["iv_tuples_raw"][index]
            i = np.copy(ivt[0])
            v = np.copy(ivt[1])
            t = np.copy(ivt[2])
        return i, v, t

    def get_i_v(
        self,
        index: int = 0,
        plain: bool = False,
    ):
        v = self.mapped["voltage_axis"]
        if plain:
            i = self.to_plot["plain"]["current"][0]
        else:
            i = self.to_plot["current"][index]
        return v, i

    def get_v_i(
        self,
        index: int = 0,
        plain: bool = False,
    ):
        i = self.mapped["current_axis"]
        if plain:
            v = self.to_plot["plain"]["voltage"][0]
        else:
            v = self.to_plot["voltage"][index]
        return i, v

    def get_didv_v(
        self,
        index: int = 0,
        plain: bool = False,
    ):
        v = self.mapped["voltage_axis"]
        if plain:
            didv = self.to_plot["plain"]["differential_conductance"][0]
        else:
            didv = self.to_plot["differential_conductance"][index]
        return v, didv

    def get_dvdi_i(
        self,
        index: int = 0,
        plain: bool = False,
    ):
        i = self.mapped["current_axis"]
        if plain:
            dvdi = self.to_plot["plain"]["differential_resistance"][0]
        else:
            dvdi = self.to_plot["differential_resistance"][index]
        return i, dvdi

    def get_T_v(
        self,
        index: int = 0,
        plain: bool = False,
    ):
        v = self.mapped["voltage_axis"]
        if plain:
            T = self.to_plot["plain"]["temperature_current"][0]
        else:
            T = self.to_plot["temperature_current"][index]
        return v, T

    def get_T_i(
        self,
        index: int = 0,
        plain: bool = False,
    ):
        v = self.mapped["current_axis"]
        if plain:
            T = self.to_plot["plain"]["temperature_voltage"][0]
        else:
            T = self.to_plot["temperature_voltage"][index]
        return v, T

    def get_didv_vy(
        self,
    ):
        v = np.copy(self.mapped["voltage_axis"])
        y = np.copy(self.mapped["y_axis"])
        didv = np.copy(self.to_plot["differential_conductance"])
        return v, y, didv

    def get_dvdi_iy(
        self,
    ):
        i = np.copy(self.mapped["current_axis"])
        y = np.copy(self.mapped["y_axis"])
        didv = np.copy(self.to_plot["differential_resistance"])
        return i, y, didv

    def get_T_vy(
        self,
    ):
        v = self.mapped["voltage_axis"]
        y = self.mapped["y_axis"]
        T = self.to_plot["temperature_current"]
        return v, y, T

    def get_T_iy(
        self,
    ):
        i = self.mapped["current_axis"]
        y = self.mapped["y_axis"]
        T = self.to_plot["temperature_voltage"]
        return i, y, T

    def get_T_y(
        self,
    ):
        y = np.copy(self.mapped["y_axis"])
        T = np.copy(self.to_plot["temperature"])
        return y, T

    # endregion

    # region get ax

    def ax_i_t_tuples(
        self,
        ax: Axes,
        index: int = 0,
        skip: list[int] = [10, -10],
        plain: bool = True,
        inverse: bool = False,
        raw: bool = False,
        i_lim: tuple[float | None, float | None] = (None, None),
        t_lim: tuple[float | None, float | None] = (None, None),
        **kwargs,
    ):
        # get data
        if raw:
            i, _, t = self.get_ivt_raw(
                index,
                plain,
                skip,
            )
        else:
            i, _, t = self.get_ivt(
                index,
                plain,
                skip,
            )
        t -= np.nanmin(t)

        # get norm and limit
        if self.i_norm is None:
            i_norm_value, i_norm_string = get_norm(i)
        else:
            i_norm_value, i_norm_string = self.i_norm
        i_lim = (
            i_lim[0] if i_lim[0] is not None else self.i_lim[0],
            i_lim[1] if i_lim[1] is not None else self.i_lim[1],
        )
        i_lim = (
            i_lim[0] / i_norm_value if i_lim[0] is not None else None,
            i_lim[1] / i_norm_value if i_lim[1] is not None else None,
        )

        if self.t_norm is None:
            t_norm_value, t_norm_string = get_norm(t)
        else:
            t_norm_value, t_norm_string = self.t_norm
        t_lim = (
            t_lim[0] if t_lim[0] is not None else self.t_lim[0],
            t_lim[1] if t_lim[1] is not None else self.t_lim[1],
        )
        t_lim = (
            t_lim[0] / t_norm_value if t_lim[0] is not None else None,
            t_lim[1] / t_norm_value if t_lim[1] is not None else None,
        )

        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", left=True, right=True, top=True, bottom=True)

        if not inverse:
            ax.set_xlabel(rf"$t$ ({t_norm_string}s)")
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(t / t_norm_value, i / i_norm_value, ".", **kwargs)
            ax.set_xlim(left=t_lim[0], right=t_lim[1])
            ax.set_ylim(bottom=i_lim[0], top=i_lim[1])
        else:
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.set_ylabel(rf"$t$ ({t_norm_string}s)")
            ax.plot(i / i_norm_value, t / t_norm_value, ".", **kwargs)
            ax.set_xlim(left=i_lim[0], right=i_lim[1])
            ax.set_ylim(bottom=t_lim[0], top=t_lim[1])
        return ax

    def ax_v_t_tuples(
        self,
        ax: Axes,
        index: int = 0,
        skip: list[int] = [10, -10],
        plain: bool = True,
        inverse: bool = False,
        raw: bool = False,
        v_lim: tuple[float | None, float | None] = (None, None),
        t_lim: tuple[float | None, float | None] = (None, None),
        **kwargs,
    ):
        # get data
        if raw:
            _, v, t = self.get_ivt_raw(
                index,
                plain,
                skip,
            )
        else:
            _, v, t = self.get_ivt(
                index,
                plain,
                skip,
            )
        t -= np.nanmin(t)

        # get norm and limit
        if self.v_norm is None:
            v_norm_value, v_norm_string = get_norm(v)
        else:
            v_norm_value, v_norm_string = self.v_norm
        v_lim = (
            v_lim[0] if v_lim[0] is not None else self.v_lim[0],
            v_lim[1] if v_lim[1] is not None else self.v_lim[1],
        )
        v_lim = (
            v_lim[0] / v_norm_value if v_lim[0] is not None else None,
            v_lim[1] / v_norm_value if v_lim[1] is not None else None,
        )

        if self.t_norm is None:
            t_norm_value, t_norm_string = get_norm(t)
        else:
            t_norm_value, t_norm_string = self.t_norm
        t_lim = (
            t_lim[0] if t_lim[0] is not None else self.t_lim[0],
            t_lim[1] if t_lim[1] is not None else self.t_lim[1],
        )
        t_lim = (
            t_lim[0] / t_norm_value if t_lim[0] is not None else None,
            t_lim[1] / t_norm_value if t_lim[1] is not None else None,
        )

        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", left=True, right=True, top=True, bottom=True)

        if not inverse:
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.set_xlabel(rf"$t$ ({t_norm_string}s)")
            ax.plot(t / t_norm_value, v / v_norm_value, ".", **kwargs)
            ax.set_xlim(left=t_lim[0], right=t_lim[1])
            ax.set_ylim(bottom=v_lim[0], top=v_lim[1])

        else:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(rf"$t$ ({t_norm_string}s)")
            ax.plot(v / v_norm_value, t / t_norm_value, ".", **kwargs)
            ax.set_xlim(left=v_lim[0], right=v_lim[1])
            ax.set_ylim(bottom=t_lim[0], top=t_lim[1])

        return ax

    def ax_v_i(
        self,
        ax: Axes,
        index: int = 0,
        skip: list[int] = [10, -10],
        plain: bool = True,
        inverse: bool = False,
        mode: str = "tuples_raw",
        v_lim: tuple[float | None, float | None] = (None, None),
        i_lim: tuple[float | None, float | None] = (None, None),
        **kwargs,
    ):
        # get data
        match mode:
            case "tuples_raw":
                i, v, _ = self.get_ivt_raw(
                    index,
                    plain,
                    skip,
                )
            case "tuples":
                i, v, _ = self.get_ivt(
                    index,
                    plain,
                    skip,
                )
            case "v_i":
                i, v = self.get_v_i(index, plain)
            case "i_v":
                v, i = self.get_i_v(index, plain)
            case _:
                logger.info(
                    "(%s) ax_v_i() mode must be: 'tuples_raw','tuples', 'i_v', v_i'",
                    self._iv_plot_name,
                )
                return

        # get norm and limit
        if self.v_norm is None:
            v_norm_value, v_norm_string = get_norm(v)
        else:
            v_norm_value, v_norm_string = self.v_norm
        v_lim = (
            v_lim[0] if v_lim[0] is not None else self.v_lim[0],
            v_lim[1] if v_lim[1] is not None else self.v_lim[1],
        )
        v_lim = (
            v_lim[0] / v_norm_value if v_lim[0] is not None else None,
            v_lim[1] / v_norm_value if v_lim[1] is not None else None,
        )

        if self.i_norm is None:
            i_norm_value, i_norm_string = get_norm(i)
        else:
            i_norm_value, i_norm_string = self.i_norm
        i_lim = (
            i_lim[0] if i_lim[0] is not None else self.i_lim[0],
            i_lim[1] if i_lim[1] is not None else self.i_lim[1],
        )
        i_lim = (
            i_lim[0] / i_norm_value if i_lim[0] is not None else None,
            i_lim[1] / i_norm_value if i_lim[1] is not None else None,
        )

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", left=True, right=True, top=True, bottom=True)

        if not inverse:
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(i / i_norm_value, v / v_norm_value, ".", **kwargs)
            ax.set_xlim(left=i_lim[0], right=i_lim[1])
            ax.set_ylim(bottom=v_lim[0], top=v_lim[1])

        else:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(v / v_norm_value, i / i_norm_value, ".", **kwargs)
            ax.set_xlim(left=v_lim[0], right=v_lim[1])
            ax.set_ylim(bottom=i_lim[0], top=i_lim[1])

        return ax

    def ax_didv_v(
        self,
        ax: Axes,
        index: int = 0,
        plain: bool = True,
        inverse: bool = False,
        v_lim: tuple[float | None, float | None] = (None, None),
        didv_lim: tuple[float | None, float | None] = (None, None),
        **kwargs,
    ):
        # get data
        v, didv = self.get_didv_v(plain=plain, index=index)

        # Apply optional smoothing to dI/dV map
        if self.smoothing:
            didv = do_smoothing(
                data=np.array([didv]),
                window_length=self.window_length,
                polyorder=self.polyorder,
            )
            didv = didv[0]

        # get norm and limit
        if self.v_norm is None:
            v_norm_value, v_norm_string = get_norm(v)
        else:
            v_norm_value, v_norm_string = self.v_norm

        v_lim = (
            v_lim[0] if v_lim[0] is not None else self.v_lim[0],
            v_lim[1] if v_lim[1] is not None else self.v_lim[1],
        )
        v_lim = (
            v_lim[0] / v_norm_value if v_lim[0] is not None else None,
            v_lim[1] / v_norm_value if v_lim[1] is not None else None,
        )

        if self.didv_norm is None:
            didv_norm_value, didv_norm_string = get_norm(didv)
        else:
            didv_norm_value, didv_norm_string = self.didv_norm
        didv_lim = (
            didv_lim[0] if didv_lim[0] is not None else self.didv_lim[0],
            didv_lim[1] if didv_lim[1] is not None else self.didv_lim[1],
        )
        didv_lim = (
            didv_lim[0] / didv_norm_value if didv_lim[0] is not None else None,
            didv_lim[1] / didv_norm_value if didv_lim[1] is not None else None,
        )

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", left=True, right=True, top=True, bottom=True)

        if not inverse:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(rf"d$I$/d$V$ ({didv_norm_string}$G_0$)")
            ax.plot(v / v_norm_value, didv / didv_norm_value, marker=".", **kwargs)
            ax.set_xlim(left=v_lim[0], right=v_lim[1])
            ax.set_ylim(bottom=didv_lim[0], top=didv_lim[1])
        else:
            ax.set_xlabel(rf"d$I$/d$V$ ({didv_norm_string}$G_0$)")
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.plot(didv / didv_norm_value, v / v_norm_value, marker=".", **kwargs)
            ax.set_xlim(left=didv_lim[0], right=didv_lim[1])
            ax.set_ylim(bottom=v_lim[0], top=v_lim[1])
        return ax

    def ax_dvdi_i(
        self,
        ax: Axes,
        index: int = 0,
        plain: bool = True,
        inverse: bool = False,
        i_lim: tuple[float | None, float | None] = (None, None),
        dvdi_lim: tuple[float | None, float | None] = (None, None),
        **kwargs,
    ):
        # get_data
        i, dvdi = self.get_dvdi_i(index, plain)

        # Apply optional smoothing to dI/dV map
        if self.smoothing:
            dvdi = do_smoothing(
                data=np.array([dvdi]),
                window_length=self.window_length,
                polyorder=self.polyorder,
            )
            dvdi = dvdi[0]

        # get norm and limit
        if self.dvdi_norm is None:
            dvdi_norm_value, dvdi_norm_string = get_norm(dvdi)
        else:
            dvdi_norm_value, dvdi_norm_string = self.dvdi_norm
        dvdi_lim = (
            dvdi_lim[0] if dvdi_lim[0] is not None else self.dvdi_lim[0],
            dvdi_lim[1] if dvdi_lim[1] is not None else self.dvdi_lim[1],
        )
        dvdi_lim = (
            dvdi_lim[0] / dvdi_norm_value if dvdi_lim[0] is not None else None,
            dvdi_lim[1] / dvdi_norm_value if dvdi_lim[1] is not None else None,
        )

        if self.i_norm is None:
            i_norm_value, i_norm_string = get_norm(i)
        else:
            i_norm_value, i_norm_string = self.i_norm
        i_lim = (
            i_lim[0] if i_lim[0] is not None else self.i_lim[0],
            i_lim[1] if i_lim[1] is not None else self.i_lim[1],
        )
        i_lim = (
            i_lim[0] / i_norm_value if i_lim[0] is not None else None,
            i_lim[1] / i_norm_value if i_lim[1] is not None else None,
        )

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", left=True, right=True, top=True, bottom=True)

        if not inverse:
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.set_ylabel(rf"d$V$/d$I$ ({dvdi_norm_string}$\Omega$)")
            ax.plot(i / i_norm_value, dvdi / dvdi_norm_value, ".", **kwargs)
            ax.set_xlim(left=i_lim[0], right=i_lim[1])
            ax.set_ylim(bottom=dvdi_lim[0], top=dvdi_lim[1])

        else:
            ax.set_xlabel(rf"d$V$/d$I$ ({dvdi_norm_string}$\Omega$)")
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(dvdi / dvdi_norm_value, i / i_norm_value, ".", **kwargs)
            ax.set_xlim(left=dvdi_lim[0], right=dvdi_lim[1])
            ax.set_ylim(bottom=i_lim[0], top=i_lim[1])
        return ax

    def ax_T_y(
        self,
        ax: Axes,
        inverse: bool = False,
        T_lim: tuple[float | None, float | None] = (None, None),
        y_lim: tuple[float | None, float | None] = (None, None),
        **kwargs,
    ):

        # get_data
        y, T = self.get_T_y()

        # get norm and limit
        if self.y_norm is None:
            y_norm_value, y_norm_string = get_norm(y)
        else:
            y_norm_value, y_norm_string = self.y_norm
        y_lim = (
            y_lim[0] if y_lim[0] is not None else self.y_lim[0],
            y_lim[1] if y_lim[1] is not None else self.y_lim[1],
        )
        y_lim = (
            y_lim[0] / y_norm_value if y_lim[0] is not None else None,
            y_lim[1] / y_norm_value if y_lim[1] is not None else None,
        )

        if self.T_norm is None:
            T_norm_value, T_norm_string = get_norm(T)
        else:
            T_norm_value, T_norm_string = self.T_norm
        T_lim = (
            T_lim[0] if T_lim[0] is not None else self.T_lim[0],
            T_lim[1] if T_lim[1] is not None else self.T_lim[1],
        )
        T_lim = (
            T_lim[0] / T_norm_value if T_lim[0] is not None else None,
            T_lim[1] / T_norm_value if T_lim[1] is not None else None,
        )

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", left=True, right=True, top=True, bottom=True)

        if not inverse:
            ax.set_xlabel(
                rf"{self.y_characters[0]} ({y_norm_string}{self.y_characters[1]})"
            )
            ax.set_ylabel(rf"$T$ ({T_norm_string}K)")
            ax.plot(y / y_norm_value, T / T_norm_value, ".", **kwargs)
            ax.set_xlim(left=y_lim[0], right=y_lim[1])
            ax.set_ylim(bottom=T_lim[0], top=T_lim[1])

        else:
            ax.set_xlabel(rf"$T$ ({T_norm_string}K)")
            ax.set_ylabel(
                rf"{self.y_characters[0]} ({y_norm_string}{self.y_characters[1]})"
            )
            ax.plot(T / T_norm_value, y / y_norm_value, ".", **kwargs)
            ax.set_xlim(left=T_lim[0], right=T_lim[1])
            ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
        return ax

    def ax_didv_vy(
        self,
        fig: Figure,
        axs: list[Axes],
        inverse: bool = False,
        cmap: ListedColormap | None = None,
        didv_c_lim: tuple[float | None, float | None] = (None, None),
        v_lim: tuple[float | None, float | None] = (None, None),
        y_lim: tuple[float | None, float | None] = (None, None),
        contrast: float | None = None,
        **kwargs,
    ):
        # Use provided axes
        ax = axs[0]
        cax = axs[1]

        # Load data: voltage (V), position (Y), differential conductance (dI/dV)
        v, y, didv = self.get_didv_vy()

        # Apply optional smoothing to dI/dV map
        if self.smoothing:
            didv = do_smoothing(
                data=didv, window_length=self.window_length, polyorder=self.polyorder
            )

        # get norm and limit
        if self.v_norm is None:
            v_norm_value, v_norm_string = get_norm(v)
        else:
            v_norm_value, v_norm_string = self.v_norm
        v_lim = (
            v_lim[0] if v_lim[0] is not None else self.v_lim[0],
            v_lim[1] if v_lim[1] is not None else self.v_lim[1],
        )
        v_lim = (
            v_lim[0] / v_norm_value if v_lim[0] is not None else None,
            v_lim[1] / v_norm_value if v_lim[1] is not None else None,
        )

        if self.y_norm is None:
            y_norm_value, y_norm_string = get_norm(y)
        else:
            y_norm_value, y_norm_string = self.y_norm
        y_lim = (
            y_lim[0] if y_lim[0] is not None else self.y_lim[0],
            y_lim[1] if y_lim[1] is not None else self.y_lim[1],
        )
        y_lim = (
            y_lim[0] / y_norm_value if y_lim[0] is not None else None,
            y_lim[1] / y_norm_value if y_lim[1] is not None else None,
        )

        if self.didv_norm is None:
            didv_norm_value, didv_norm_string = get_norm(didv)
        else:
            didv_norm_value, didv_norm_string = self.didv_norm
        didv_c_lim = (
            didv_c_lim[0] if didv_c_lim[0] is not None else self.didv_c_lim[0],
            didv_c_lim[1] if didv_c_lim[1] is not None else self.didv_c_lim[1],
        )
        didv_c_lim = (
            didv_c_lim[0] / didv_norm_value if didv_c_lim[0] is not None else None,
            didv_c_lim[1] / didv_norm_value if didv_c_lim[1] is not None else None,
        )

        # Determine color limits and plotting extent
        if contrast is None:
            contrast = self.contrast
        didv_c_lim = get_z_lim(didv / didv_norm_value, didv_c_lim, contrast)
        ext, _, _ = get_ext(x=v / v_norm_value, y=y / y_norm_value)

        # Configure axes based on orientation
        if not inverse:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(
                rf"{self.y_characters[0]} ({y_norm_string}{self.y_characters[1]})"
            )
            orientation = "vertical"
        else:
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.set_xlabel(
                rf"{self.y_characters[0]} ({y_norm_string}{self.y_characters[1]})"
            )
            didv = np.rot90(didv)
            orientation = "horizontal"
            ext = [ext[2], ext[3], ext[0], ext[1]]

        if cmap is None:
            cmap = self.cmap
        # Plot the image with correct color scaling and axis extents
        im = ax.imshow(
            didv / didv_norm_value,
            cmap=cmap,
            extent=tuple(ext),
            clim=didv_c_lim,
            aspect="auto",
            origin="lower",
            interpolation="none",
            **kwargs,
        )

        # Add colorbar with proper label and orientation
        fig.colorbar(
            im,
            cax=cax,
            label=rf"d$I$/d$V$ ({didv_norm_string}$G_0$)",
            orientation=orientation,
        )

        # Tweak ticks and style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", left=True, right=True, top=True, bottom=True)

        return ax, cax

    def ax_dvdi_iy(
        self,
        fig: Figure,
        axs: list[Axes],
        inverse: bool = False,
        cmap: ListedColormap | None = None,
        dvdi_c_lim: tuple[float | None, float | None] = (None, None),
        i_lim: tuple[float | None, float | None] = (None, None),
        y_lim: tuple[float | None, float | None] = (None, None),
        contrast: float = 1,
        **kwargs,
    ):
        # Use provided axes
        ax = axs[0]
        cax = axs[1]

        # Load data: voltage (V), position (Y), differential resistance (dV/dI)
        i, y, dvdi = self.get_dvdi_iy()

        # Apply optional smoothing to dI/dV map
        if self.smoothing:
            dvdi = do_smoothing(
                data=dvdi, window_length=self.window_length, polyorder=self.polyorder
            )

        # Normalize axes and values for nicer labels and scaling
        if self.y_norm is None:
            y_norm_value, y_norm_string = get_norm(y)
        else:
            y_norm_value, y_norm_string = self.y_norm
        y_lim = (
            y_lim[0] if y_lim[0] is not None else self.y_lim[0],
            y_lim[1] if y_lim[1] is not None else self.y_lim[1],
        )
        y_lim = (
            y_lim[0] / y_norm_value if y_lim[0] is not None else None,
            y_lim[1] / y_norm_value if y_lim[1] is not None else None,
        )

        if self.i_norm is None:
            i_norm_value, i_norm_string = get_norm(i)
        else:
            i_norm_value, i_norm_string = self.i_norm
        i_lim = (
            i_lim[0] if i_lim[0] is not None else self.i_lim[0],
            i_lim[1] if i_lim[1] is not None else self.i_lim[1],
        )
        i_lim = (
            i_lim[0] / i_norm_value if i_lim[0] is not None else None,
            i_lim[1] / i_norm_value if i_lim[1] is not None else None,
        )

        if self.dvdi_norm is None:
            dvdi_norm_value, dvdi_norm_string = get_norm(dvdi)
        else:
            dvdi_norm_value, dvdi_norm_string = self.dvdi_norm

        dvdi_c_lim = (
            dvdi_c_lim[0] if dvdi_c_lim[0] is not None else self.dvdi_c_lim[0],
            dvdi_c_lim[1] if dvdi_c_lim[1] is not None else self.dvdi_c_lim[1],
        )
        dvdi_c_lim = (
            dvdi_c_lim[0] / dvdi_norm_value if dvdi_c_lim[0] is not None else None,
            dvdi_c_lim[1] / dvdi_norm_value if dvdi_c_lim[1] is not None else None,
        )

        # Determine color limits and plotting extent
        dvdi_c_lim = get_z_lim(dvdi / dvdi_norm_value, dvdi_c_lim, contrast)
        ext, _, _ = get_ext(x=i / i_norm_value, y=y / y_norm_value)

        # Configure axes based on orientation
        if not inverse:
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.set_ylabel(
                rf"{self.y_characters[0]} ({y_norm_string}{self.y_characters[1]})"
            )
            orientation = "vertical"
        else:
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.set_xlabel(
                rf"{self.y_characters[0]} ({y_norm_string}{self.y_characters[1]})"
            )
            dvdi = np.rot90(dvdi)
            orientation = "horizontal"
            ext = [ext[2], ext[3], ext[0], ext[1]]

        if cmap is None:
            cmap = self.cmap
        # Plot the image with correct color scaling and axis extents
        im = ax.imshow(
            dvdi / dvdi_norm_value,
            cmap=cmap,
            extent=tuple(ext),
            clim=dvdi_c_lim,
            aspect="auto",
            origin="lower",
            interpolation="none",
            **kwargs,
        )

        # Add colorbar with proper label and orientation
        fig.colorbar(
            im,
            cax=cax,
            label=rf"d$V$/d$I$ ({dvdi_norm_string}$\Omega$)",
            orientation=orientation,
        )

        # Tweak ticks and style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", left=True, right=True, top=True, bottom=True)

        return ax, cax

    # endregion

    # region get fig

    def fig_didv_vy(
        self,
        fig_nr: int = 0,
        width_ratios: list[float] = [4, 0.2],
        cmap: ListedColormap | None = None,
        v_lim: tuple[float | None, float | None] = (None, None),
        y_lim: tuple[float | None, float | None] = (None, None),
        didv_c_lim: tuple[float | None, float | None] = (None, None),
        contrast: float = 1,
    ):
        plt.close(fig_nr)
        fig, axs = plt.subplots(
            num=fig_nr,
            nrows=1,
            ncols=2,
            figsize=self.fig_size,
            dpi=self.dpi,
            gridspec_kw={"width_ratios": width_ratios},
            constrained_layout=True,
        )
        # get axs
        axs = self.ax_didv_vy(
            fig=fig,
            axs=axs,
            cmap=cmap,
            v_lim=v_lim,
            y_lim=y_lim,
            didv_c_lim=didv_c_lim,
            contrast=contrast,
        )

        return fig, axs

    def fig_didv_vy_T(
        self,
        fig_nr: int = 0,
        width_ratios: list[float] = [1.0, 4.0, 0.2],
        cmap: ListedColormap | None = None,
        v_lim: tuple[float | None, float | None] = (None, None),
        y_lim: tuple[float | None, float | None] = (None, None),
        T_lim: tuple[float | None, float | None] = (None, None),
        didv_c_lim: tuple[float | None, float | None] = (None, None),
        contrast: float = 1,
    ):
        plt.close(fig_nr)
        fig, axs = plt.subplots(
            num=fig_nr,
            nrows=1,
            ncols=3,
            figsize=self.fig_size,
            dpi=self.dpi,
            gridspec_kw={"width_ratios": width_ratios},
            constrained_layout=True,
        )
        # get axs
        axs[1], axs[2] = self.ax_didv_vy(
            fig=fig,
            axs=[axs[1], axs[2]],
            cmap=cmap,
            v_lim=v_lim,
            y_lim=y_lim,
            didv_c_lim=didv_c_lim,
            contrast=contrast,
        )
        axs[0] = self.ax_T_y(ax=axs[0], inverse=True, linestyle="", color="grey")

        # modify
        axs[0].xaxis.set_inverted(True)
        axs[0].sharey(axs[1])
        plt.setp(axs[1].get_yticklabels(), visible=False)
        axs[1].set_ylabel("")

        # set limits
        axs[0].set_xlim(T_lim[::-1])

        return fig, axs

    def fig_dvdi_iy(
        self,
        fig_nr: int = 0,
        width_ratios: list[float] = [4, 0.2],
        cmap: ListedColormap | None = None,
        i_lim: tuple[float | None, float | None] = (None, None),
        y_lim: tuple[float | None, float | None] = (None, None),
        dvdi_c_lim: tuple[float | None, float | None] = (None, None),
        contrast: float = 1,
    ):
        plt.close(fig_nr)
        fig, axs = plt.subplots(
            num=fig_nr,
            nrows=1,
            ncols=2,
            figsize=self.fig_size,
            dpi=self.dpi,
            gridspec_kw={"width_ratios": width_ratios},
            constrained_layout=True,
        )
        # get axs
        axs = self.ax_dvdi_iy(
            fig=fig,
            axs=axs,
            cmap=cmap,
            i_lim=i_lim,
            y_lim=y_lim,
            dvdi_c_lim=dvdi_c_lim,
            contrast=contrast,
        )

        return fig, axs

    def fig_dvdi_iy_T(
        self,
        fig_nr: int = 0,
        width_ratios: list[float] = [1.0, 4.0, 0.2],
        cmap: ListedColormap | None = None,
        i_lim: tuple[float | None, float | None] = (None, None),
        y_lim: tuple[float | None, float | None] = (None, None),
        T_lim: tuple[float | None, float | None] = (None, None),
        dvdi_c_lim: tuple[float | None, float | None] = (None, None),
        contrast: float = 1,
    ):
        plt.close(fig_nr)
        fig, axs = plt.subplots(
            num=fig_nr,
            nrows=1,
            ncols=3,
            figsize=self.fig_size,
            dpi=self.dpi,
            gridspec_kw={"width_ratios": width_ratios},
            constrained_layout=True,
        )
        # get axs
        axs[1], axs[2] = self.ax_dvdi_iy(
            fig=fig,
            axs=[axs[1], axs[2]],
            cmap=cmap,
            i_lim=i_lim,
            y_lim=y_lim,
            dvdi_c_lim=dvdi_c_lim,
            contrast=contrast,
        )
        axs[0] = self.ax_T_y(ax=axs[0], inverse=True, linestyle="", color="grey")

        # modify
        axs[0].xaxis.set_inverted(True)
        axs[0].sharey(axs[1])
        plt.setp(axs[1].get_yticklabels(), visible=False)
        axs[1].set_ylabel("")

        # set limits
        axs[0].set_xlim(T_lim[::-1])

        return fig, axs

    def fig_ididv_v(
        self,
        fig_nr: int = 0,
        v_lim: tuple = (None, None),
        i_lim: tuple = (None, None),
        didv_lim: tuple = (None, None),
        dvdi_lim: tuple = (None, None),
        plain: bool = True,
        index: int = 0,
    ):
        plt.close(fig_nr)
        fig, axs = plt.subplots(
            num=fig_nr,
            nrows=2,
            ncols=2,
            figsize=self.fig_size,
            dpi=self.dpi,
            gridspec_kw={"height_ratios": [1, 3], "width_ratios": [1, 3]},
            constrained_layout=True,
        )

        axs[0, 0].remove()

        axs[1, 1] = self.ax_v_i(
            ax=axs[1, 1],
            mode="tuples_raw",
            index=index,
            plain=plain,
            inverse=True,
            color="lightgrey",
            label="raw",
            linestyle="-",
        )

        axs[1, 1] = self.ax_v_i(
            ax=axs[1, 1],
            mode="tuples",
            index=index,
            plain=plain,
            inverse=True,
            color="grey",
            label="sampled",
        )
        axs[1, 1] = self.ax_v_i(
            ax=axs[1, 1],
            mode="i_v",
            index=index,
            plain=plain,
            inverse=True,
            color="red",
            label="I(V)",
        )
        axs[1, 1] = self.ax_v_i(
            ax=axs[1, 1],
            mode="v_i",
            index=index,
            plain=plain,
            inverse=True,
            color="blue",
            label="V(I)",
        )
        axs[1, 1].plot([0, 0], [0, 0], "white", label="T(K)", zorder=0)
        T = self.get_T(index=index, plain=plain)
        axs[1, 1].legend(
            ["raw", "sampled", "$I(V)$", "$V(I)$", f"$T={T*1000:06.1f}$mK"],
            fontsize=8,
            loc="upper left",
        )

        axs[0, 1] = self.ax_didv_v(
            ax=axs[0, 1], index=index, plain=plain, inverse=False
        )
        axs[1, 0] = self.ax_dvdi_i(ax=axs[1, 0], index=index, plain=plain, inverse=True)

        # modify
        axs[1, 0].xaxis.set_inverted(True)

        axs[0, 1].xaxis.set_label_position("top")
        axs[0, 1].xaxis.tick_top()
        axs[0, 1].tick_params(bottom=True)
        axs[0, 1].yaxis.set_label_position("right")
        axs[0, 1].yaxis.tick_right()
        axs[0, 1].tick_params(left=True)

        axs[1, 1].yaxis.set_label_position("right")
        axs[1, 1].yaxis.tick_right()
        axs[1, 1].tick_params(left=True)

        # set limits
        axs[1, 1].set_xlim(v_lim)
        axs[1, 1].set_ylim(i_lim)
        axs[0, 1].set_ylim(didv_lim)
        axs[1, 0].set_xlim(dvdi_lim[::-1])

        axs[1, 1].sharex(axs[0, 1])
        axs[1, 1].sharey(axs[1, 0])

        return fig, axs

    # endregion

    # begin plot all

    def plot_all(
        self,
        y_lim: tuple[float | None, float | None] = (None, None),
        v_lim: tuple[float | None, float | None] = (None, None),
        i_lim: tuple[float | None, float | None] = (None, None),
        T_lim: tuple[float | None, float | None] = (None, None),
        didv_lim: tuple[float | None, float | None] = (None, None),
        dvdi_lim: tuple[float | None, float | None] = (None, None),
        didv_c_lim: tuple[float | None, float | None] = (None, None),
        dvdi_c_lim: tuple[float | None, float | None] = (None, None),
    ):

        if self.plot_didvs:
            if self.plot_T:
                fig, axs = self.fig_didv_vy_T(
                    v_lim=v_lim,
                    y_lim=y_lim,
                    didv_c_lim=didv_c_lim,
                    T_lim=T_lim,
                    fig_nr=self.plot_index,
                )
                fig.suptitle(
                    f"{self.sub_folder}/{self.title}/{self.title_of_plot}/didv_vy_T"
                )
                self.saveFigure(
                    fig, sub_title="didv_vy_T", sub_folder=self.title_of_plot
                )
                self.plot_index += 1
            else:
                fig, axs = self.fig_didv_vy(
                    v_lim=v_lim,
                    y_lim=y_lim,
                    didv_c_lim=didv_c_lim,
                    fig_nr=self.plot_index,
                )
                fig.suptitle(
                    f"{self.sub_folder}/{self.title}/{self.title_of_plot}/didv_vy"
                )
                self.saveFigure(fig, sub_title="didv_vy", sub_folder=self.title_of_plot)
                self.plot_index += 1

        if self.plot_dvdis:
            if self.plot_T:
                fig, axs = self.fig_dvdi_iy_T(
                    i_lim=i_lim,
                    y_lim=y_lim,
                    dvdi_c_lim=dvdi_c_lim,
                    T_lim=T_lim,
                    fig_nr=self.plot_index,
                )
                fig.suptitle(
                    f"{self.sub_folder}/{self.title}/{self.title_of_plot}/dvdi_iy_T"
                )
                self.saveFigure(
                    fig, sub_title="dvdi_iy_T", sub_folder=self.title_of_plot
                )
                self.plot_index += 1
            else:
                fig, axs = self.fig_dvdi_iy(
                    i_lim=i_lim,
                    y_lim=y_lim,
                    dvdi_c_lim=dvdi_c_lim,
                    fig_nr=self.plot_index,
                )
                fig.suptitle(
                    f"{self.sub_folder}/{self.title}/{self.title_of_plot}/dvdi_iy"
                )
                self.saveFigure(fig, sub_title="dvdi_iy", sub_folder=self.title_of_plot)
                self.plot_index += 1

        if self.plot_ivs:
            logger.info("(%s) saveIVs()", self._iv_plot_name)
            with suppress_logging():
                try:
                    fig, ax = self.fig_ididv_v(
                        v_lim=v_lim,
                        i_lim=i_lim,
                        didv_lim=didv_lim,
                        dvdi_lim=dvdi_lim,
                        fig_nr=self.plot_index,
                        plain=True,
                    )
                    fig.suptitle(
                        f"{self.sub_folder}/{self.title}/{self.title_of_plot}/IV/0000_y=NAN",
                        fontsize=8,
                    )
                    self.saveFigure(
                        fig,
                        sub_title=f"0000_y=NAN",
                        sub_folder=f"{self.title_of_plot}/IV",
                    )
                    self.plot_index += 1
                except KeyError:
                    logger.warning(
                        "(%s) No plain data available for %s",
                        self._iv_plot_name,
                        self.title_of_plot,
                    )
                    pass

                for j, y in enumerate(tqdm(self.mapped["y_axis"])):
                    fig, ax = self.fig_ididv_v(
                        v_lim=v_lim,
                        i_lim=i_lim,
                        didv_lim=didv_lim,
                        dvdi_lim=dvdi_lim,
                        fig_nr=self.plot_index,
                        plain=False,
                        index=j,
                    )
                    fig.suptitle(
                        f"{self.sub_folder}/{self.title}/{self.title_of_plot}/IV/{j+1:04d}_y={y:05.3f}",
                        fontsize=8,
                    )
                    self.saveFigure(
                        fig,
                        sub_title=f"{j+1:04d}_y={y:05.3f}",
                        sub_folder=f"{self.title_of_plot}/IV",
                    )
                    self.plot_index += 1

    # endregion

    # region properties

    @property
    def smoothing(self):
        """get smoothing"""
        return self.iv_plot["smoothing"]

    @smoothing.setter
    def smoothing(self, smoothing: bool):
        """set smoothing"""
        self.iv_plot["smoothing"] = smoothing
        logger.info("(%s) smoothing = %s", self._iv_plot_name, smoothing)

    @property
    def window_length(self):
        """get window_length"""
        return self.iv_plot["window_length"]

    @window_length.setter
    def window_length(self, window_length: int):
        """set window_length"""
        self.iv_plot["window_length"] = window_length
        logger.info("(%s) window_length = %s", self._iv_plot_name, window_length)

    @property
    def polyorder(self):
        """get polyorder"""
        return self.iv_plot["polyorder"]

    @polyorder.setter
    def polyorder(self, polyorder: int):
        """set polyorder"""
        self.iv_plot["polyorder"] = polyorder
        logger.info("(%s) polyorder = %s", self._iv_plot_name, polyorder)

    @property
    def cmap(self):
        """get cmap"""
        return self.iv_plot["cmap"]

    @cmap.setter
    def cmap(self, cmap):
        """set cmap"""
        self.iv_plot["cmap"] = cmap
        logger.info("(%s) cmap = %s", self._iv_plot_name, cmap)

    @property
    def fig_size(self):
        """get fig_size"""
        return self.iv_plot["fig_size"]

    @fig_size.setter
    def fig_size(self, fig_size: tuple[float]):
        """set fig_size"""
        self.iv_plot["fig_size"] = fig_size
        logger.info("(%s) fig_size = %s", self._iv_plot_name, fig_size)

    @property
    def dpi(self):
        """get dpi"""
        return self.iv_plot["dpi"]

    @dpi.setter
    def dpi(self, dpi: int):
        """set dpi"""
        self.iv_plot["dpi"] = dpi
        logger.info("(%s) dpi = %s", self._iv_plot_name, dpi)

    # endregion
