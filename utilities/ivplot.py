# region imports
import sys
import logging
import importlib

import matplotlib.pyplot as plt
import numpy as np


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

# endregion


# region functions
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
        super().__init__()
        self._iv_plot_name = name

        self.to_plot = self.get_empty_dictionary()
        self.title_of_plot = ""
        self.y_characters = ["$y$", ""]

        self.iv_plot = {
            "fig_size": (6, 4),
            "dpi": 100,
            "cmap": cmap(bad="grey"),
            "smoothing": True,
            "window_length": 20,
            "polyorder": 2,
        }

        logger.info("(%s) ... BasePlot initialized.", self._iv_plot_name)

    # region get data

    def get_ivt(
        self,
        index: int = 0,
        plain: bool = False,
        skip: list[int] = [10, -10],
    ):
        if plain:
            ivt = self.to_plot["plain"]["iv_tuples"][0]
            i = ivt[0][skip[0] : skip[1]]
            v = ivt[1][skip[0] : skip[1]]
            t = ivt[2][skip[0] : skip[1]]
        else:
            ivt = self.to_plot["iv_tuples"][index]
            i = ivt[0]
            v = ivt[1]
            t = ivt[2]
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
        ax=None,
        index: int = 0,
        skip: list[int] = [10, -10],
        plain: bool = False,
        fig_nr: int = 0,
        inverse: bool = False,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get data
        i, _, t = self.get_ivt(
            index,
            plain,
            skip,
        )
        t -= np.nanmin(t)

        # get_norm
        i_norm_value, i_norm_string = get_norm(i)
        t_norm_value, t_norm_string = get_norm(t)

        i /= i_norm_value
        t /= t_norm_value

        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.set_xlabel(rf"$t$ ({t_norm_string}s)")
            ax.plot(t, i, ".")

        else:
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.set_ylabel(rf"$t$ ({t_norm_string}s)")
            ax.plot(i, t, ".")
        return ax

    def ax_v_t_tuples(
        self,
        ax=None,
        index: int = 0,
        skip: list[int] = [10, -10],
        plain: bool = False,
        fig_nr: int = 0,
        inverse: bool = False,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get data
        _, v, t = self.get_ivt(
            index,
            plain,
            skip,
        )
        t -= np.nanmin(t)

        # get_norm
        v_norm_value, v_norm_string = get_norm(v)
        t_norm_value, t_norm_string = get_norm(t)

        v /= v_norm_value
        t /= t_norm_value

        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.set_xlabel(rf"$t$ ({t_norm_string}s)")
            ax.plot(t, v, ".")

        else:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(rf"$t$ ({t_norm_string}s)")
            ax.plot(v, t, ".")

        return ax

    def ax_v_i_tuples(
        self,
        ax=None,
        index: int = 0,
        skip: list[int] = [10, -10],
        plain: bool = False,
        fig_nr: int = 0,
        inverse: bool = False,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get data
        i, v, _ = self.get_ivt(
            index,
            plain,
            skip,
        )

        # get_norm
        i_norm_value, i_norm_string = get_norm(i)
        v_norm_value, v_norm_string = get_norm(v)

        i /= i_norm_value
        v /= v_norm_value

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(i, v, ".")

        else:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(v, i, ".")

        return ax

    def ax_i_v(
        self,
        ax=None,
        index: int = 0,
        plain: bool = False,
        fig_nr: int = 0,
        inverse: bool = False,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get_data
        v, i = self.get_i_v(index, plain)

        # get_norm
        v_norm_value, v_norm_string = get_norm(v)
        i_norm_value, i_norm_string = get_norm(i)

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(v / v_norm_value, i / i_norm_value, ".")

        else:
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.plot(i / i_norm_value, v / v_norm_value, ".")
        return ax

    def ax_v_i(
        self,
        ax=None,
        index: int = 0,
        plain: bool = False,
        fig_nr: int = 0,
        inverse: bool = False,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get_data
        i, v = self.get_v_i(index, plain)

        # get_norm
        v_norm_value, v_norm_string = get_norm(v)
        i_norm_value, i_norm_string = get_norm(i)

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.plot(i / i_norm_value, v / v_norm_value, ".")

        else:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(v / v_norm_value, i / i_norm_value, ".")
        return ax

    def ax_didv_v(
        self,
        ax=None,
        index: int = 0,
        plain: bool = False,
        fig_nr: int = 0,
        inverse: bool = False,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get_data
        v, didv = self.get_didv_v(index, plain)

        # get_norm
        v_norm_value, v_norm_string = get_norm(v)
        didv_norm_value, didv_norm_string = get_norm(didv)

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(rf"d$I$/d$V$ ({didv_norm_string}$G_0$)")
            ax.plot(v / v_norm_value, didv / didv_norm_value, ".")

        else:
            ax.set_xlabel(rf"d$I$/d$V$ ({didv_norm_string}$G_0$)")
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.plot(didv / didv_norm_value, v / v_norm_value, ".")
        return ax

    def ax_dvdi_i(
        self,
        ax=None,
        index: int = 0,
        plain: bool = False,
        fig_nr: int = 0,
        inverse: bool = False,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get_data
        i, dvdi = self.get_dvdi_i(index, plain)

        # get_norm
        i_norm_value, i_norm_string = get_norm(i)
        dvdi_norm_value, dvdi_norm_string = get_norm(dvdi)

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.set_ylabel(rf"d$V$/d$I$ ({dvdi_norm_string}$\Omega$)")
            ax.plot(i / i_norm_value, dvdi / dvdi_norm_value, ".")

        else:
            ax.set_xlabel(rf"d$V$/d$I$ ({dvdi_norm_string}$\Omega$)")
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(dvdi / dvdi_norm_value, i / i_norm_value, ".")
        return ax

    def ax_T_v(
        self,
        ax=None,
        index: int = 0,
        plain: bool = False,
        fig_nr: int = 0,
        inverse: bool = False,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get_data
        v, T = self.get_T_v(index, plain)

        # get_norm
        v_norm_value, v_norm_string = get_norm(v)
        T_norm_value, T_norm_string = get_norm(T)

        v /= v_norm_value
        T /= T_norm_value

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_xlabel(rf"$V$ ({v_norm_string}V)")
            ax.set_ylabel(rf"$T$ ({T_norm_string}K)")
            ax.plot(v / v_norm_value, T / T_norm_value, ".")

        else:
            ax.set_xlabel(rf"$T$ ({T_norm_string}K)")
            ax.set_ylabel(rf"$V$ ({v_norm_string}V)")
            ax.plot(T, v, ".")
        return ax

    def ax_T_i(
        self,
        ax=None,
        index: int = 0,
        plain: bool = False,
        fig_nr: int = 0,
        inverse: bool = False,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get_data
        i, T = self.get_T_i(index, plain)

        # get_norm
        i_norm_value, i_norm_string = get_norm(i)
        T_norm_value, T_norm_string = get_norm(T)

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_xlabel(rf"$I$ ({i_norm_string}A)")
            ax.set_ylabel(rf"$T$ ({T_norm_string}K)")
            ax.plot(i / i_norm_value, T / T_norm_value, ".")

        else:
            ax.set_xlabel(rf"$T$ ({T_norm_string}K)")
            ax.set_ylabel(rf"$I$ ({i_norm_string}A)")
            ax.plot(T / T_norm_value, i / i_norm_value, ".")
        return ax

    def ax_didv_vy(
        self,
        faxs: list = [],
        fig_nr: int = 0,
        inverse: bool = False,
        z_lim: tuple = (None, None),
        z_contrast: float = 1,
        **kwargs,
    ):
        """
        Plot dI/dV as a 2D color map over V and Y.

        Parameters
        ----------
        faxs : list, optional
            If provided, should contain [fig, ax, cax] to plot on existing axes.
        fig_nr : int, optional
            Figure number for matplotlib (if new figure is created).
        inverse : bool, optional
            If True, swap X and Y axes in the plot.
        z_lim : tuple, optional
            Z-axis (color) limits. If None, will be auto-calculated.
        z_contrast : float, optional
            Factor to scale standard deviation for z-lim if not provided.
        **kwargs : dict
            Additional keyword arguments passed to `imshow`.
        """
        if not faxs:
            # Create new figure and axis layout if no axes passed
            plt.close(fig_nr)
            fig, [ax, cax] = plt.subplots(num=fig_nr, ncols=2)
        else:
            # Use provided axes
            fig = faxs[0]
            ax = faxs[1]
            cax = faxs[2]

        # Load data: voltage (V), position (Y), differential conductance (dI/dV)
        v, y, didv = self.get_didv_vy()

        # Apply optional smoothing to dI/dV map
        if self.smoothing:
            didv = do_smoothing(
                data=didv, window_length=self.window_length, polyorder=self.polyorder
            )

        # Normalize axes and values for nicer labels and scaling
        v_norm_value, v_norm_string = get_norm(v)
        y_norm_value, y_norm_string = get_norm(y)
        didv_norm_value, didv_norm_string = get_norm(didv)

        v /= v_norm_value
        y /= y_norm_value
        didv /= didv_norm_value

        # Determine color limits and plotting extent
        z_lim = get_z_lim(didv, z_lim, z_contrast)
        z_ext, _, _ = get_ext(x=v, y=y)

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
            z_ext = [z_ext[2], z_ext[3], z_ext[0], z_ext[1]]

        # Plot the image with correct color scaling and axis extents
        im = ax.imshow(
            didv,
            cmap=self.cmap,
            extent=z_ext,
            clim=z_lim,
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
        ax.tick_params(direction="in", right=True, top=True)

        return ax, cax

    def ax_dvdi_iy(
        self,
        faxs: list = [],
        fig_nr: int = 0,
        inverse: bool = False,
        z_lim: tuple = (None, None),
        z_contrast: float = 1,
        **kwargs,
    ):
        """
        Plot dI/dV as a 2D color map over V and Y.

        Parameters
        ----------
        faxs : list, optional
            If provided, should contain [fig, ax, cax] to plot on existing axes.
        fig_nr : int, optional
            Figure number for matplotlib (if new figure is created).
        inverse : bool, optional
            If True, swap X and Y axes in the plot.
        z_lim : tuple, optional
            Z-axis (color) limits. If None, will be auto-calculated.
        z_contrast : float, optional
            Factor to scale standard deviation for z-lim if not provided.
        **kwargs : dict
            Additional keyword arguments passed to `imshow`.
        """
        if not faxs:
            # Create new figure and axis layout if no axes passed
            plt.close(fig_nr)
            fig, [ax, cax] = plt.subplots(num=fig_nr, ncols=2)
        else:
            # Use provided axes
            fig = faxs[0]
            ax = faxs[1]
            cax = faxs[2]

        # Load data: voltage (V), position (Y), differential conductance (dI/dV)
        i, y, dvdi = self.get_dvdi_iy()

        # Apply optional smoothing to dI/dV map
        if self.smoothing:
            dvdi = do_smoothing(
                data=dvdi, window_length=self.window_length, polyorder=self.polyorder
            )

        # Normalize axes and values for nicer labels and scaling
        i_norm_value, i_norm_string = get_norm(i)
        y_norm_value, y_norm_string = get_norm(y)
        dvdi_norm_value, dvdi_norm_string = get_norm(dvdi)

        i /= i_norm_value
        y /= y_norm_value
        dvdi /= dvdi_norm_value

        # Determine color limits and plotting extent
        z_lim = get_z_lim(dvdi, z_lim, z_contrast)
        z_ext, _, _ = get_ext(x=i, y=y)

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
            z_ext = [z_ext[2], z_ext[3], z_ext[0], z_ext[1]]

        # Plot the image with correct color scaling and axis extents
        im = ax.imshow(
            dvdi,
            cmap=self.cmap,
            extent=z_ext,
            clim=z_lim,
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
        ax.tick_params(direction="in", right=True, top=True)

        return ax, cax

    def ax_T_vy(
        self,
        faxs: list = [],
        fig_nr: int = 0,
        inverse: bool = False,
        z_lim: tuple = (None, None),
        z_contrast: float = 1,
        **kwargs,
    ):
        if not faxs:
            # Create new figure and axis layout if no axes passed
            plt.close(fig_nr)
            fig, [ax, cax] = plt.subplots(num=fig_nr, ncols=2)
        else:
            # Use provided axes
            fig = faxs[0]
            ax = faxs[1]
            cax = faxs[2]

        # Load data: voltage (V), position (Y), differential conductance (dI/dV)
        v, y, T = self.get_T_vy()

        # Normalize axes and values for nicer labels and scaling
        v_norm_value, v_norm_string = get_norm(v)
        y_norm_value, y_norm_string = get_norm(y)
        T_norm_value, didv_norm_string = get_norm(T)

        v /= v_norm_value
        y /= y_norm_value
        T /= T_norm_value

        # Determine color limits and plotting extent
        z_lim = get_z_lim(T, z_lim, z_contrast)
        z_ext, _, _ = get_ext(x=v, y=y)

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
            T = np.rot90(T)
            orientation = "horizontal"
            z_ext = [z_ext[2], z_ext[3], z_ext[0], z_ext[1]]

        # Plot the image with correct color scaling and axis extents
        im = ax.imshow(
            T,
            cmap=self.cmap,
            extent=z_ext,
            clim=z_lim,
            aspect="auto",
            origin="lower",
            interpolation="none",
            **kwargs,
        )

        # Add colorbar with proper label and orientation
        fig.colorbar(
            im,
            cax=cax,
            label=rf"$T$ ({didv_norm_string}K)",
            orientation=orientation,
        )

        # Tweak ticks and style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        return ax, cax

    def ax_T_y(
        self,
        ax=None,
        fig_nr: int = 0,
        inverse: bool = False,
        **kwargs,
    ):
        if not ax:
            # Generate Figure
            plt.close(fig_nr)
            fig, ax = plt.subplots(num=fig_nr)

        # get_data
        y, T = self.get_T_y()

        # get_norm
        y_norm_value, y_norm_string = get_norm(y)
        T_norm_value, T_norm_string = get_norm(T)

        y /= y_norm_value
        T /= T_norm_value

        # set tick style
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-9, 9), useMathText=True)
        ax.tick_params(direction="in", right=True, top=True)

        if not inverse:
            ax.set_xlabel(
                rf"{self.y_characters[0]} ({y_norm_string}{self.y_characters[1]})"
            )
            ax.set_ylabel(rf"$T$ ({T_norm_string}K)")
            ax.plot(y, T, **kwargs)

        else:
            ax.set_xlabel(rf"$T$ ({T_norm_string}K)")
            ax.set_ylabel(
                rf"{self.y_characters[0]} ({y_norm_string}{self.y_characters[1]})"
            )
            ax.plot(T, y, **kwargs)
        return ax

    # endregion

    # region get fig

    def fig_didv_vy(
        self,
        fig_nr: int = 0,
        x_lim: tuple = (None, None),
        y_lim: tuple = (None, None),
        z_lim: tuple = (None, None),
    ):
        plt.close(fig_nr)
        fig, axs = plt.subplots(
            num=fig_nr,
            nrows=1,
            ncols=2,
            figsize=self.fig_size,
            dpi=self.dpi,
            gridspec_kw={"width_ratios": [4, 0.2]},
            constrained_layout=True,
        )
        # get axs
        axs[0], axs[1] = self.ax_didv_vy(faxs=[fig, axs[0], axs[1]], z_lim=z_lim)

        # set limits
        axs[0].set_xlim(x_lim)
        axs[0].set_ylim(y_lim)

        return fig, axs

    def fig_didv_vy_T(
        self,
        fig_nr: int = 0,
        x_lim: tuple = (None, None),
        y_lim: tuple = (None, None),
        z_lim: tuple = (None, None),
    ):
        plt.close(fig_nr)
        fig, axs = plt.subplots(
            num=fig_nr,
            nrows=1,
            ncols=3,
            figsize=self.fig_size,
            dpi=self.dpi,
            gridspec_kw={"width_ratios": [1, 4, 0.2]},
            constrained_layout=True,
        )
        # get axs
        axs[1], axs[2] = self.ax_didv_vy(faxs=[fig, axs[1], axs[2]], z_lim=z_lim)
        axs[0] = self.ax_T_y(
            ax=axs[0], inverse=True, marker=".", linestyle="", color="grey"
        )

        # modify
        axs[0].xaxis.set_inverted(True)
        axs[0].sharey(axs[1])
        plt.setp(axs[1].get_yticklabels(), visible=False)
        axs[1].set_ylabel("")

        # set limits
        axs[1].set_xlim(x_lim)
        axs[1].set_ylim(y_lim)

        return fig, axs

    def fig_dvdi_iy(
        self,
        fig_nr: int = 0,
        x_lim: tuple = (None, None),
        y_lim: tuple = (None, None),
        z_lim: tuple = (None, None),
    ):
        plt.close(fig_nr)
        fig, axs = plt.subplots(
            num=fig_nr,
            nrows=1,
            ncols=2,
            figsize=self.fig_size,
            dpi=self.dpi,
            gridspec_kw={"width_ratios": [4, 0.2]},
            constrained_layout=True,
        )
        # get axs
        axs[0], axs[1] = self.ax_dvdi_iy(faxs=[fig, axs[0], axs[1]], z_lim=z_lim)

        # set limits
        axs[0].set_xlim(x_lim)
        axs[0].set_ylim(y_lim)

        return fig, axs

    def fig_dvdi_iy_T(
        self,
        fig_nr: int = 0,
        x_lim: tuple = (None, None),
        y_lim: tuple = (None, None),
        z_lim: tuple = (None, None),
    ):
        plt.close(fig_nr)
        fig, axs = plt.subplots(
            num=fig_nr,
            nrows=1,
            ncols=3,
            figsize=self.fig_size,
            dpi=self.dpi,
            gridspec_kw={"width_ratios": [1, 4, 0.2]},
            constrained_layout=True,
        )
        # get axs
        axs[1], axs[2] = self.ax_dvdi_iy(faxs=[fig, axs[1], axs[2]], z_lim=z_lim)
        axs[0] = self.ax_T_y(
            ax=axs[0], inverse=True, marker=".", linestyle="", color="grey"
        )

        # modify
        axs[0].xaxis.set_inverted(True)
        axs[0].sharey(axs[1])
        plt.setp(axs[1].get_yticklabels(), visible=False)
        axs[1].set_ylabel("")

        # set limits
        axs[0].set_xlim(x_lim)
        axs[0].set_ylim(y_lim)

        return fig, axs

    def fig_ivs(
        self,
        index: int = 0,
        plain: bool = False,
        fig_nr: int = 0,
        x_lim: tuple = (None, None),
        y_lim: tuple = (None, None),
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        plt.close(fig_nr)
        fig, ax = plt.subplots(
            num=fig_nr,
            nrows=1,
            ncols=1,
            figsize=self.fig_size,
            dpi=self.dpi,
            constrained_layout=True,
        )

        (line,) = ax.plot([], [], "r.")

        i, v, t = self.get_ivt(index=0)
        # get_norm
        v_norm_value, v_norm_string = get_norm(v)
        i_norm_value, i_norm_string = get_norm(i)

        # Initialization function: plot empty frame
        def init():
            line.set_data(v / v_norm_value, i / i_norm_value)
            return (line,)

        # set limits
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        def update(frame):
            i, v, _ = self.get_ivt(index=frame)
            line.set_data(v / v_norm_value, i / i_norm_value)
            return (line,)

        frames = np.arange(self.mapped["y_axis"].shape[0])

        ani = FuncAnimation(
            fig, update, frames=frames, init_func=init, blit=False, interval=20
        )
        plt.show()
        ani.save("sine_wave.mp4", fps=30)

        return fig, ax

    # endregion

    # begin plot all

    def plot_all(self, leading_index: int = 0):
        i = 0
        fig, axs = self.fig_didv_vy(z_lim=(None, None), fig_nr=leading_index + i)

        fig.suptitle(f"{self.sub_folder}/{self.title}/{self.title_of_plot}/didv_vy")

        self.saveFigure(fig, sub_title="didv_vy", sub_folder=self.title_of_plot)

        i += 1

        fig, axs = self.fig_didv_vy_T(z_lim=(None, None), fig_nr=leading_index + i)

        fig.suptitle(f"{self.sub_folder}/{self.title}/{self.title_of_plot}/didv_vy_T")

        self.saveFigure(fig, sub_title="didv_vy_T", sub_folder=self.title_of_plot)

        i += 1

        fig, axs = self.fig_dvdi_iy(z_lim=(None, None), fig_nr=leading_index + i)

        fig.suptitle(f"{self.sub_folder}/{self.title}/{self.title_of_plot}/dvdi_iy")

        self.saveFigure(fig, sub_title="dvdi_iy", sub_folder=self.title_of_plot)

        i += 1

        fig, axs = self.fig_dvdi_iy_T(z_lim=(None, None), fig_nr=leading_index + i)

        fig.suptitle(f"{self.sub_folder}/{self.title}/{self.title_of_plot}/dvdi_iy_T")

        self.saveFigure(fig, sub_title="dvdi_iy_T", sub_folder=self.title_of_plot)

        i += 1

        # get axs
        for index, y_value in enumerate(self.mapped["y_axis"]):
            print(index, y_value)

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
