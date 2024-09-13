"""
Description.

"""

import os
import sys
import logging
import importlib

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import savgol_filter

from utilities.baseevaluation import BaseEvaluation
from utilities.corporate_design_colors_v4 import cmap
from utilities.baseplotting_functions import PLOT_KEYS
from utilities.baseplotting_functions import plot_test_map
from utilities.baseplotting_functions import plot_map
from utilities.baseplotting_functions import plot_map_vector
from utilities.baseplotting_functions import plot_iv

importlib.reload(sys.modules["utilities.baseplotting_functions"])
importlib.reload(sys.modules["utilities.baseevaluation"])
importlib.reload(sys.modules["utilities.corporate_design_colors_v4"])

logger = logging.getLogger(__name__)


class BasePlotting(BaseEvaluation):
    """
    Description
    """

    def __init__(
        self,
        name="base plot",
    ):
        """
        Description
        Keep in mind that x_lim, etc. the first entry should be smaller than the second.
        None is also possible.
        """
        super().__init__(name=name)

        self.possible_plot_keys = PLOT_KEYS

        self.plot = {
            "inverted": False,
            "smoothing": True,
            "window_length": 20,
            "polyorder": 2,
            "fig_nr_show_map": 0,
            "fig_nr_show_map_vector": 1,
            "fig_nr_IV": 2,
            "display_dpi": 100,
            "png_dpi": 600,
            "pdf_dpi": 600,
            "save_pdf": False,
            "color_map": cmap(color="seeblau", bad="gray"),
            "contrast": None,
            "vector_color": None,
            "vector_style": ".",
            "vector_lwms": 1.5,
            "x_lim": [None, None],
            "y_lim": [None, None],
            "z_lim": [None, None],
            "n_lim": [None, None],
            "i_lim": [None, None],
            "x_key": "V_bias_up_mV",
            "y_key": "y_axis",
            "z_key": "dIdV_up",
            "n_key": "T_up_K",
            "i_key": "I_up_A",
        }

        self.show_map = {}
        self.show_map_vector = {}
        self.show_iv = {}
        logger.info("(%s) ... BasePlotting initialized.", self._name)

    def saveFigure(
        self,
        figure,
        subtitle,
    ):
        """saveFigure()
        - safes Figure to self.figure_folder/self.title
        """

        # Handle Title
        title = f"{self.base['title']}"

        # Handle data folder
        folder = os.path.join(
            os.getcwd(),
            self.base["figure_folder"],
            self.base["sub_folder"],
            self.base["title"],
        )
        check = os.path.isdir(folder)
        if not check:
            os.makedirs(folder)

        # Save Everything
        name = os.path.join(folder, subtitle)
        figure.savefig(f"{name}.png", dpi=self.plot["png_dpi"])
        if self.plot["save_pdf"]:  # save as pdf
            logger.info(
                "(%s) saveFigure() to %s%s.pdf",
                self._name,
                self.base["figure_folder"],
                self.base["title"],
            )
            figure.savefig(f"{name}.pdf", dpi=self.plot["pdf_dpi"])

    def showMap(
        self,
    ):
        """showMap()
        - checks for synthax errors
        - get data and label from plot_keys
        - calls plot_map()

        Parameters
        ----------
        x_key : str = 'V_bias_up_mV'
            select plot_key_x from self.plot_keys
        y_key : str = 'y_axis'
            select plot_key_y from self.plot_keys
        z_key : str = 'dIdV_up'
            select plot_key_z from self.plot_keys
        x_lim : list = [np.nan, np.nan]
            sets limits on x-Axis
        y_lim : list = [np.nan, np.nan]
            sets limits on y-Axis
        z_lim : list = [np.nan, np.nan]
            sets limits on z-Axis / colorbar
        """

        logger.info(
            "(%s) showMap('%s', '%s', '%s')",
            self._name,
            self.plot["x_key"],
            self.plot["y_key"],
            self.plot["z_key"],
        )

        warning = False

        try:
            plot_key_x = self.possible_plot_keys[self.plot["x_key"]]
        except KeyError:
            logger.warning("(%s) x_key not found.", self._name)
            warning = True

        try:
            plot_key_y = self.possible_plot_keys[self.plot["y_key"]]
        except KeyError:
            logger.warning("(%s) y_key not found.", self._name)
            warning = True

        try:
            plot_key_z = self.possible_plot_keys[self.plot["z_key"]]
        except KeyError:
            logger.warning("(%s) z_key not found.", self._name)
            warning = True

        if warning:
            plot_test_map()
            return

        try:
            x_data = eval(plot_key_x[0])  # pylint: disable=eval-used
            y_data = eval(plot_key_y[0])  # pylint: disable=eval-used
            z_data = eval(plot_key_z[0])  # pylint: disable=eval-used
        except AttributeError:
            logger.warning(
                "(%s) Required data not found. Check if data is calculated and plot_keys!",
                self._name,
            )
            return

        if self.plot["smoothing"]:
            z_data = savgol_filter(
                z_data,
                window_length=self.plot["window_length"],
                polyorder=self.plot["polyorder"],
            )

        x_label = plot_key_x[1]
        y_label = plot_key_y[1]
        z_label = plot_key_z[1]

        fig, ax_z, ax_c = plot_map(
            x=x_data,
            y=y_data,
            z=z_data,
            x_lim=self.plot["x_lim"],
            y_lim=self.plot["y_lim"],
            z_lim=self.plot["z_lim"],
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            inverted=self.plot["inverted"],
            fig_nr=self.plot["fig_nr_show_map"],
            color_map=self.plot["color_map"],
            display_dpi=self.plot["display_dpi"],
            contrast=self.plot["contrast"],
        )
        fig.suptitle(self.base["title"])

        self.show_map = {
            "title": self.base["title"],
            "fig": fig,
            "ax_z": ax_z,
            "ax_c": ax_c,
            "x_key": self.plot["x_key"],
            "y_key": self.plot["y_key"],
            "z_key": self.plot["z_key"],
            "x_lim": self.plot["x_lim"],
            "y_lim": self.plot["y_lim"],
            "z_lim": self.plot["z_lim"],
            "x_data": x_data,
            "y_data": y_data,
            "z_data": z_data,
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
            "inverted": self.plot["inverted"],
            "fig_nr": self.plot["fig_nr_show_map"],
            "color_map": self.plot["color_map"],
            "display_dpi": self.plot["display_dpi"],
            "contrast": self.plot["contrast"],
        }
        plt.show()

    def reshowMap(
        self,
    ):
        """reshowMap()
        - shows Figure
        """
        logger.info(
            "(%s) reshowMap()",
            self._name,
        )
        if self.show_map:
            plot_map(
                x=self.show_map["x_data"],
                y=self.show_map["y_data"],
                z=self.show_map["z_data"],
                x_lim=self.show_map["x_lim"],
                y_lim=self.show_map["y_lim"],
                z_lim=self.show_map["z_lim"],
                x_label=self.show_map["x_label"],
                y_label=self.show_map["y_label"],
                z_label=self.show_map["z_label"],
                inverted=self.show_map["inverted"],
                fig_nr=self.show_map["fig_nr"],
                color_map=self.show_map["color_map"],
                display_dpi=self.show_map["display_dpi"],
                contrast=self.show_map["contrast"],
            )
            plt.suptitle(self.show_map["title"])
            plt.show()

    def saveMap(self):
        """saveMap()
        - safes Figure to self.figure_folder/self.title
        """
        logger.info(
            "(%s) saveMap() to %s%s.png",
            self._name,
            self.base["figure_folder"],
            self.base["title"],
        )
        string = "Map"
        if self.show_map["inverted"]:
            string += "Inverted"
        self.saveFigure(figure=self.show_map["fig"], subtitle=string)

    def showMapVector(
        self,
    ):
        """
        - calls plot_map_vector()

        """

        logger.info(
            "(%s) showMapVector('%s', '%s', '%s', '%s')",
            self._name,
            self.plot["x_key"],
            self.plot["y_key"],
            self.plot["z_key"],
            self.plot["n_key"],
        )

        warning = False

        try:
            plot_key_x = self.possible_plot_keys[self.plot["x_key"]]
        except KeyError:
            logger.warning("(%s) x_key not found.", self._name)
            warning = True

        try:
            plot_key_y = self.possible_plot_keys[self.plot["y_key"]]
        except KeyError:
            logger.warning("(%s) y_key not found.", self._name)
            warning = True

        try:
            plot_key_z = self.possible_plot_keys[self.plot["z_key"]]
        except KeyError:
            logger.warning("(%s) z_key not found.", self._name)
            warning = True

        try:
            plot_key_n = self.possible_plot_keys[self.plot["n_key"]]
        except KeyError:
            logger.warning("(%s) n_key not found.", self._name)
            warning = True

        if warning:
            plot_test_map()
            return

        try:
            x_data = eval(plot_key_x[0])  # pylint: disable=eval-used
            y_data = eval(plot_key_y[0])  # pylint: disable=eval-used
            z_data = eval(plot_key_z[0])  # pylint: disable=eval-used
            n_data = eval(plot_key_n[0])  # pylint: disable=eval-used
        except AttributeError:
            logger.warning(
                "(%s) Required data not found. Check if data is calculated and plot_keys!",
                self._name,
            )
            return

        if self.plot["smoothing"]:
            z_data = savgol_filter(
                z_data,
                window_length=self.plot["window_length"],
                polyorder=self.plot["polyorder"],
            )

        x_label = plot_key_x[1]
        y_label = plot_key_y[1]
        z_label = plot_key_z[1]
        n_label = plot_key_n[1]

        fig, ax_z, ax_c, ax_n = plot_map_vector(
            x=x_data,
            y=y_data,
            z=z_data,
            n=n_data,
            x_lim=self.plot["x_lim"],
            y_lim=self.plot["y_lim"],
            z_lim=self.plot["z_lim"],
            n_lim=self.plot["n_lim"],
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            n_label=n_label,
            inverted=self.plot["inverted"],
            fig_nr=self.plot["fig_nr_show_map_vector"],
            color_map=self.plot["color_map"],
            display_dpi=self.plot["display_dpi"],
            contrast=self.plot["contrast"],
            vector_color=self.plot["vector_color"],
            vector_style=self.plot["vector_style"],
            vector_lwms=self.plot["vector_lwms"],
        )
        fig.suptitle(self.base["title"])

        self.show_map_vector = {
            "title": self.base["title"],
            "fig": fig,
            "ax_z": ax_z,
            "ax_c": ax_c,
            "ax_n": ax_n,
            "x_key": self.plot["x_key"],
            "y_key": self.plot["y_key"],
            "z_key": self.plot["z_key"],
            "n_key": self.plot["n_key"],
            "x_lim": self.plot["x_lim"],
            "y_lim": self.plot["y_lim"],
            "z_lim": self.plot["z_lim"],
            "n_lim": self.plot["n_lim"],
            "x_data": x_data,
            "y_data": y_data,
            "z_data": z_data,
            "n_data": n_data,
            "x_label": x_label,
            "y_label": y_label,
            "z_label": z_label,
            "n_label": n_label,
            "inverted": self.plot["inverted"],
            "fig_nr": self.plot["fig_nr_show_map_vector"],
            "color_map": self.plot["color_map"],
            "display_dpi": self.plot["display_dpi"],
            "contrast": self.plot["contrast"],
            "vector_color": self.plot["vector_color"],
            "vector_style": self.plot["vector_style"],
            "vector_lwms": self.plot["vector_lwms"],
        }
        plt.show()

    def reshowMapVector(
        self,
    ):
        """reshowMapVector()
        - reshows MapVector
        """
        logger.info(
            "(%s) reshowMapVector()",
            self._name,
        )
        if self.show_map_vector:
            plot_map_vector(
                x=self.show_map_vector["x_data"],
                y=self.show_map_vector["y_data"],
                z=self.show_map_vector["z_data"],
                n=self.show_map_vector["n_data"],
                x_lim=self.show_map_vector["x_lim"],
                y_lim=self.show_map_vector["y_lim"],
                z_lim=self.show_map_vector["z_lim"],
                n_lim=self.show_map_vector["n_lim"],
                x_label=self.show_map_vector["x_label"],
                y_label=self.show_map_vector["y_label"],
                z_label=self.show_map_vector["z_label"],
                n_label=self.show_map_vector["n_label"],
                inverted=self.show_map_vector["inverted"],
                fig_nr=self.show_map_vector["fig_nr"],
                color_map=self.show_map_vector["color_map"],
                display_dpi=self.show_map_vector["display_dpi"],
                contrast=self.show_map_vector["contrast"],
                vector_color=self.show_map_vector["vector_color"],
                vector_style=self.show_map_vector["vector_style"],
                vector_lwms=self.show_map_vector["vector_lwms"],
            )
            plt.suptitle(self.show_map_vector["title"])
            plt.show()

    def saveMapVector(self):
        """saveMap()
        - safes Figure to self.figure_folder/self.title
        """
        logger.info(
            "(%s) saveMapVector() to %s%s.png",
            self._name,
            self.base["figure_folder"],
            self.base["title"],
        )
        string = "MapVector"
        if self.show_map_vector["inverted"]:
            string += "Inverted"
        self.saveFigure(figure=self.show_map_vector["fig"], subtitle=string)

    def showIV(
        self,
        indices=None,
        values=None,
    ):
        """showIV()"""
        logger.info(
            "(%s) showIV('%s', '%s', '%s', '%s', '%s')",
            self._name,
            self.plot["x_key"],
            self.plot["y_key"],
            self.plot["z_key"],
            self.plot["n_key"],
            self.plot["i_key"],
        )
        ppk = self.possible_plot_keys
        x_data = eval(ppk[self.plot["x_key"]][0])  # pylint: disable=eval-used
        y_data = eval(ppk[self.plot["y_key"]][0])  # pylint: disable=eval-used
        z_data = eval(ppk[self.plot["z_key"]][0])  # pylint: disable=eval-used
        n_data = eval(ppk[self.plot["n_key"]][0])  # pylint: disable=eval-used
        i_data = eval(ppk[self.plot["i_key"]][0])  # pylint: disable=eval-used

        if self.plot["smoothing"]:
            z_data = savgol_filter(
                z_data,
                window_length=self.plot["window_length"],
                polyorder=self.plot["polyorder"],
            )
            i_data = savgol_filter(
                i_data,
                window_length=self.plot["window_length"],
                polyorder=self.plot["polyorder"],
            )

        if indices is None:
            indices = []
        if values is not None:
            for value in values:
                indices.append(int(np.argmin(np.abs(y_data - value))))
        if indices == []:
            indices.append(0)

        fig, ax_i, ax_didv, ax_y = plot_iv(
            indices=indices,
            x=x_data,
            y=y_data,
            z=z_data,
            n=n_data,
            i=i_data,
            x_lim=self.plot["x_lim"],
            y_lim=self.plot["y_lim"],
            z_lim=self.plot["z_lim"],
            n_lim=self.plot["n_lim"],
            i_lim=self.plot["i_lim"],
            x_label=ppk[self.plot["x_key"]][1],
            y_label=ppk[self.plot["y_key"]][1],
            z_label=ppk[self.plot["z_key"]][1],
            n_label=ppk[self.plot["n_key"]][1],
            i_label=ppk[self.plot["i_key"]][1],
            fig_nr=self.plot["fig_nr_IV"],
            display_dpi=self.plot["display_dpi"],
            vector_color=self.plot["vector_color"],
            vector_style=self.plot["vector_style"],
            vector_lwms=self.plot["vector_lwms"],
        )
        fig.suptitle(self.base["title"])

        self.show_iv = {
            "title": self.base["title"],
            "indices": indices,
            "fig": fig,
            "ax_i": ax_i,
            "ax_didv": ax_didv,
            "ax_y": ax_y,
            "x_key": self.plot["x_key"],
            "y_key": self.plot["y_key"],
            "z_key": self.plot["z_key"],
            "n_key": self.plot["n_key"],
            "i_key": self.plot["i_key"],
            "x_lim": self.plot["x_lim"],
            "y_lim": self.plot["y_lim"],
            "z_lim": self.plot["z_lim"],
            "n_lim": self.plot["n_lim"],
            "i_lim": self.plot["i_lim"],
            "x_data": x_data,
            "y_data": y_data,
            "z_data": z_data,
            "n_data": n_data,
            "i_data": i_data,
            "x_label": ppk[self.plot["x_key"]][1],
            "y_label": ppk[self.plot["y_key"]][1],
            "z_label": ppk[self.plot["z_key"]][1],
            "n_label": ppk[self.plot["n_key"]][1],
            "i_label": ppk[self.plot["i_key"]][1],
            "fig_nr": self.plot["fig_nr_show_map_vector"],
            "display_dpi": self.plot["display_dpi"],
            "vector_color": self.plot["vector_color"],
            "vector_style": self.plot["vector_style"],
            "vector_lwms": self.plot["vector_lwms"],
        }
        plt.show()

    def reshowIV(
        self,
    ):
        """reshowIV()
        - reshows IV
        """
        logger.info(
            "(%s) reshowIV()",
            self._name,
        )
        plot_iv(
            indices=self.show_iv["indices"],
            x=self.show_iv["x_data"],
            y=self.show_iv["y_data"],
            z=self.show_iv["z_data"],
            n=self.show_iv["n_data"],
            i=self.show_iv["i_data"],
            x_lim=self.show_iv["x_lim"],
            y_lim=self.show_iv["y_lim"],
            z_lim=self.show_iv["z_lim"],
            n_lim=self.show_iv["n_lim"],
            i_lim=self.show_iv["i_lim"],
            x_label=self.show_iv["x_label"],
            y_label=self.show_iv["y_label"],
            z_label=self.show_iv["z_label"],
            n_label=self.show_iv["n_label"],
            i_label=self.show_iv["i_label"],
            fig_nr=self.show_iv["fig_nr"],
            display_dpi=self.show_iv["display_dpi"],
            vector_color=self.show_iv["vector_color"],
            vector_style=self.show_iv["vector_style"],
            vector_lwms=self.show_iv["vector_lwms"],
        )
        plt.suptitle(self.show_iv["title"])
        plt.show()

    def saveIV(self):
        """saveIV()
        - safes Figure to self.figure_folder/self.title
        """
        logger.info(
            "(%s) saveIV() to %s%s.png",
            self._name,
            self.base["figure_folder"],
            self.base["title"],
        )
        string = f"IV {self.show_iv['indices']}"
        self.saveFigure(figure=self.show_iv["fig"], subtitle=string)

    @property
    def x_key(self):
        """get x_key"""
        return self.plot["x_key"]

    @x_key.setter
    def x_key(self, x_key: str):
        """set x_key"""
        self.plot["x_key"] = x_key
        logger.info("(%s) x_key = %s", self._name, x_key)

    @property
    def y_key(self):
        """get y_key"""
        return self.plot["y_key"]

    @y_key.setter
    def y_key(self, y_key: str):
        """set y_key"""
        self.plot["y_key"] = y_key
        logger.info("(%s) y_key = %s", self._name, y_key)

    @property
    def z_key(self):
        """get z_key"""
        return self.plot["z_key"]

    @z_key.setter
    def z_key(self, z_key: str):
        """set z_key"""
        self.plot["z_key"] = z_key
        logger.info("(%s) z_key = %s", self._name, z_key)

    @property
    def n_key(self):
        """get n_key"""
        return self.plot["n_key"]

    @n_key.setter
    def n_key(self, n_key: str):
        """set n_key"""
        self.plot["n_key"] = n_key
        logger.info("(%s) n_key = %s", self._name, n_key)

    @property
    def i_key(self):
        """get i_key"""
        return self.plot["i_key"]

    @i_key.setter
    def i_key(self, i_key: str):
        """set i_key"""
        self.plot["i_key"] = i_key
        logger.info("(%s) i_key = %s", self._name, i_key)

    @property
    def x_lim(self):
        """get x_lim"""
        return self.plot["x_lim"]

    @x_lim.setter
    def x_lim(self, x_lim: list):
        """set x_lim"""
        self.plot["x_lim"] = x_lim
        logger.info("(%s) x_lim = %s", self._name, x_lim)

    @property
    def y_lim(self):
        """get y_lim"""
        return self.plot["y_lim"]

    @y_lim.setter
    def y_lim(self, y_lim: list):
        """set y_lim"""
        self.plot["y_lim"] = y_lim
        logger.info("(%s) y_lim = %s", self._name, y_lim)

    @property
    def z_lim(self):
        """get z_lim"""
        return self.plot["z_lim"]

    @z_lim.setter
    def z_lim(self, z_lim: list):
        """set z_lim"""
        self.plot["z_lim"] = z_lim
        logger.info("(%s) z_lim = %s", self._name, z_lim)

    @property
    def n_lim(self):
        """get n_lim"""
        return self.plot["n_lim"]

    @n_lim.setter
    def n_lim(self, n_lim: list):
        """set n_lim"""
        self.plot["n_lim"] = n_lim
        logger.info("(%s) n_lim = %s", self._name, n_lim)

    @property
    def i_lim(self):
        """get i_lim"""
        return self.plot["i_lim"]

    @i_lim.setter
    def i_lim(self, i_lim: list):
        """set i_lim"""
        self.plot["i_lim"] = i_lim
        logger.info("(%s) i_lim = %s", self._name, i_lim)

    @property
    def inverted(self):
        """get inverted"""
        return self.plot["inverted"]

    @inverted.setter
    def inverted(self, inverted: bool):
        """set inverted"""
        self.plot["inverted"] = inverted
        logger.info("(%s) inverted = %s", self._name, inverted)

    @property
    def smoothing(self):
        """get smoothing"""
        return self.plot["smoothing"]

    @smoothing.setter
    def smoothing(self, smoothing: bool):
        """set smoothing"""
        self.plot["smoothing"] = smoothing
        logger.info("(%s) smoothing = %s", self._name, smoothing)

    @property
    def window_length(self):
        """get window_length"""
        return self.plot["window_length"]

    @window_length.setter
    def window_length(self, window_length: int):
        """set window_length"""
        self.plot["window_length"] = window_length
        logger.info("(%s) window_length = %s", self._name, window_length)

    @property
    def polyorder(self):
        """get polyorder"""
        return self.plot["polyorder"]

    @polyorder.setter
    def polyorder(self, polyorder: int):
        """set polyorder"""
        self.plot["polyorder"] = polyorder
        logger.info("(%s) polyorder = %s", self._name, polyorder)

    @property
    def fig_nr_show_map(self):
        """get fig_nr_show_map"""
        return self.plot["fig_nr_show_map"]

    @fig_nr_show_map.setter
    def fig_nr_show_map(self, fig_nr_show_map: int):
        """set fig_nr_show_map"""
        self.plot["fig_nr_show_map"] = fig_nr_show_map
        logger.info("(%s) fig_nr_show_map = %s", self._name, fig_nr_show_map)

    @property
    def fig_nr_show_map_vector(self):
        """get fig_nr_show_map_vector"""
        return self.plot["fig_nr_show_map_vector"]

    @fig_nr_show_map_vector.setter
    def fig_nr_show_map_vector(self, fig_nr_show_map_vector: int):
        """set fig_nr_show_map_vector"""
        self.plot["fig_nr_show_map_vector"] = fig_nr_show_map_vector
        logger.info(
            "(%s) fig_nr_show_map_vector = %s", self._name, fig_nr_show_map_vector
        )

    @property
    def fig_nr_iv(self):
        """get fig_nr_iv"""
        return self.plot["fig_nr_iv"]

    @fig_nr_iv.setter
    def fig_nr_iv(self, fig_nr_iv: int):
        """set fig_nr_iv"""
        self.plot["fig_nr_iv"] = fig_nr_iv
        logger.info("(%s) fig_nr_iv = %s", self._name, fig_nr_iv)

    @property
    def display_dpi(self):
        """get display_dpi"""
        return self.plot["display_dpi"]

    @display_dpi.setter
    def display_dpi(self, display_dpi: int):
        """set display_dpi"""
        self.plot["display_dpi"] = display_dpi
        logger.info("(%s) display_dpi = %s", self._name, display_dpi)

    @property
    def png_dpi(self):
        """get png_dpi"""
        return self.plot["png_dpi"]

    @png_dpi.setter
    def png_dpi(self, png_dpi: int):
        """set png_dpi"""
        self.plot["png_dpi"] = png_dpi
        logger.info("(%s) png_dpi = %s", self._name, png_dpi)

    @property
    def pdf_dpi(self):
        """get pdf_dpi"""
        return self.plot["pdf_dpi"]

    @pdf_dpi.setter
    def pdf_dpi(self, pdf_dpi: int):
        """set pdf_dpi"""
        self.plot["pdf_dpi"] = pdf_dpi
        logger.info("(%s) pdf_dpi = %s", self._name, pdf_dpi)

    @property
    def save_pdf(self):
        """get save_pdf"""
        return self.plot["save_pdf"]

    @save_pdf.setter
    def save_pdf(self, save_pdf: bool):
        """set save_pdf"""
        self.plot["save_pdf"] = save_pdf
        logger.info("(%s) save_pdf = %s", self._name, save_pdf)

    @property
    def color_map(self):
        """get color_map"""
        return self.plot["color_map"]

    @color_map.setter
    def color_map(self, color_map):
        """set color_map"""
        self.plot["color_map"] = color_map
        logger.info("(%s) color_map = %s", self._name, color_map)

    @property
    def contrast(self):
        """get contrast"""
        return self.plot["contrast"]

    @contrast.setter
    def contrast(self, contrast: float):
        """set contrast"""
        self.plot["contrast"] = contrast
        logger.info("(%s) contrast = %s", self._name, contrast)

    @property
    def vector_color(self):
        """get vector_color"""
        return self.plot["vector_color"]

    @vector_color.setter
    def vector_color(self, vector_color):
        """set vector_color"""
        self.plot["vector_color"] = vector_color
        logger.info("(%s) vector_color = %s", self._name, vector_color)

    @property
    def vector_style(self):
        """get vector_style"""
        return self.plot["vector_style"]

    @vector_style.setter
    def vector_style(self, vector_style: str):
        """set vector_style"""
        self.plot["vector_style"] = vector_style
        logger.info("(%s) vector_style = %s", self._name, vector_style)

    @property
    def vector_lwms(self):
        """get vector_lwms"""
        return self.plot["vector_lwms"]

    @vector_lwms.setter
    def vector_lwms(self, vector_lwms: float):
        """set vector_lwms"""
        self.plot["vector_lwms"] = vector_lwms
        logger.info("(%s) vector_lwms = %s", self._name, vector_lwms)
