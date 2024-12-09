"""
Description.

"""

import os
from pickle import TUPLE1
import sys
import logging
import importlib

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from utilities.baseevaluation import BaseEvaluation
from utilities.corporate_design_colors_v4 import cmap
from utilities.plot_all import plot_all
from utilities.key_database import PLOT_KEYS


importlib.reload(sys.modules["utilities.baseevaluation"])
importlib.reload(sys.modules["utilities.corporate_design_colors_v4"])
importlib.reload(sys.modules["utilities.plot_all"])
importlib.reload(sys.modules["utilities.key_database"])

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
            "x_key": "V_bias_up_mV",
            "y_key": "y_axis",
            "z_key": "dIdV_up",
            "v_key": "V_bias_up_mV",
            "i_key": "I_up_nA_abs",
            "didv_key": "dIdV_up",
            "n_key": "T_up_K",
            "x_lim": (None, None),
            "y_lim": (None, None),
            "z_lim": (None, None),
            "v_lim": None,
            "i_lim": (None, None),
            "didv_lim": (None, None),
            "n_lim": (None, None),
            "smoothing": True,
            "window_length": 20,
            "polyorder": 2,
            "z_cmap": cmap(color="seeblau", bad="gray"),
            "z_contrast": 0.05,
            "n_show": True,
            "n_size": (0.3, 0.3),
            "n_labelsize": 8,
            "n_color": "grey",
            "n_style": ".",
            "n_lwms": 1.5,
            "fig_nr": 0,
            "fig_size": (6, 4),
            "display_dpi": 150,
            "png_dpi": 600,
            "pdf_dpi": 600,
            "save_pdf": False,
            "indices": None,
            "values": None,
        }

        self.show_map = {}

        logger.info("(%s) ... BasePlotting initialized.", self._name)

    def saveAll(self, string=None):
        """saveIV()
        - safes Figure to self.figure_folder/self.title
        """
        logger.info(
            "(%s) saveAll() to %s%s",
            self._name,
            self.base["figure_folder"],
            self.base["title"],
        )
        if string is None:
            string = f"overview {self.show_all['indices']}"
        self.saveFigure(figure=self.show_all["fig"], subtitle=string)

    def saveFigure(
        self,
        figure,
        subtitle,
        subfolder=None,
    ):
        """saveFigure()
        - safes Figure to self.figure_folder/self.sub_folder/suptitle
        """

        # Handle data folder
        if subfolder is None:
            folder = os.path.join(
                os.getcwd(),
                self.base["figure_folder"],
                self.base["sub_folder"],
                self.base["title"],
            )
        else:
            folder = os.path.join(
                os.getcwd(),
                self.base["figure_folder"],
                self.base["sub_folder"],
                self.base["title"],
                subfolder,
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

    def getIndices(self, y_data):
        """Docstring"""
        if self.indices is None:
            indices = []
        else:
            indices = self.indices
        if self.values is not None:
            for value in self.values:
                indices.append(int(np.argmin(np.abs(y_data - value))))
        if indices == []:
            indices.append(0)
        return indices

    def showAll(
        self,
    ):
        """
        - calls plot_all()

        """
        logger.info(
            "(%s) showAll()",
            self._name,
        )

        ppk = self.possible_plot_keys
        x_data = eval(ppk[self.plot["x_key"]][0])
        y_data = eval(ppk[self.plot["y_key"]][0])
        z_data = eval(ppk[self.plot["z_key"]][0])
        v_data = eval(ppk[self.plot["v_key"]][0])
        i_data = eval(ppk[self.plot["i_key"]][0])
        didv_data = eval(ppk[self.plot["didv_key"]][0])
        n_data = eval(ppk[self.plot["n_key"]][0])

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
            didv_data = savgol_filter(
                didv_data,
                window_length=self.plot["window_length"],
                polyorder=self.plot["polyorder"],
            )

        indices = self.getIndices(y_data=y_data)

        fig, ax_z, ax_c, ax_n, ax_i, ax_didv = plot_all(
            indices=indices,
            x=x_data,
            y=y_data,
            z=z_data,
            v=v_data,
            i=i_data,
            didv=didv_data,
            n=n_data,
            x_lim=self.plot["x_lim"],
            y_lim=self.plot["y_lim"],
            z_lim=self.plot["z_lim"],
            v_lim=self.plot["v_lim"],
            i_lim=self.plot["i_lim"],
            didv_lim=self.plot["didv_lim"],
            n_lim=self.plot["n_lim"],
            x_label=ppk[self.plot["x_key"]][1],
            y_label=ppk[self.plot["y_key"]][1],
            z_label=ppk[self.plot["z_key"]][1],
            v_label=ppk[self.plot["v_key"]][1],
            i_label=ppk[self.plot["i_key"]][1],
            didv_label=ppk[self.plot["didv_key"]][1],
            n_label=ppk[self.plot["n_key"]][1],
            fig_nr=self.plot["fig_nr"],
            fig_size=self.plot["fig_size"],
            display_dpi=self.plot["display_dpi"],
            z_cmap=self.plot["z_cmap"],
            z_contrast=self.plot["z_contrast"],
            n_show=self.plot["n_show"],
            n_size=self.plot["n_size"],
            n_labelsize=self.plot["n_labelsize"],
            n_color=self.plot["n_color"],
            n_style=self.plot["n_style"],
            n_lwms=self.plot["n_lwms"],
            title=self.base["title"],
        )
        plt.show()

        self.show_all = {
            "title": self.base["title"],
            "fig": fig,
            "ax_z": ax_z,
            "ax_c": ax_c,
            "ax_n": ax_n,
            "ax_i": ax_i,
            "ax_didv": ax_didv,
            "indices": indices,
        }

    # Properties

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
    def v_key(self):
        """get v_key"""
        return self.plot["v_key"]

    @v_key.setter
    def v_key(self, v_key: str):
        """set v_key"""
        self.plot["v_key"] = v_key
        logger.info("(%s) v_key = %s", self._name, v_key)

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
    def didv_key(self):
        """get didv_key"""
        return self.plot["didv_key"]

    @didv_key.setter
    def didv_key(self, didv_key: str):
        """set didv_key"""
        self.plot["didv_key"] = didv_key
        logger.info("(%s) didv_key = %s", self._name, didv_key)

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
    def x_lim(self):
        """get x_lim"""
        return self.plot["x_lim"]

    @x_lim.setter
    def x_lim(self, x_lim: tuple):
        """set x_lim"""
        self.plot["x_lim"] = x_lim
        logger.info("(%s) x_lim = %s", self._name, x_lim)

    @property
    def y_lim(self):
        """get y_lim"""
        return self.plot["y_lim"]

    @y_lim.setter
    def y_lim(self, y_lim: tuple):
        """set y_lim"""
        self.plot["y_lim"] = y_lim
        logger.info("(%s) y_lim = %s", self._name, y_lim)

    @property
    def z_lim(self):
        """get z_lim"""
        return self.plot["z_lim"]

    @z_lim.setter
    def z_lim(self, z_lim: tuple):
        """set z_lim"""
        self.plot["z_lim"] = z_lim
        logger.info("(%s) z_lim = %s", self._name, z_lim)

    @property
    def i_lim(self):
        """get i_lim"""
        return self.plot["i_lim"]

    @i_lim.setter
    def i_lim(self, i_lim: tuple):
        """set i_lim"""
        self.plot["i_lim"] = i_lim
        logger.info("(%s) i_lim = %s", self._name, i_lim)

    @property
    def v_lim(self):
        """get v_lim"""
        return self.plot["v_lim"]

    @v_lim.setter
    def v_lim(self, v_lim: float):
        """set v_lim"""
        self.plot["v_lim"] = v_lim
        logger.info("(%s) v_lim = %s", self._name, v_lim)

    @property
    def didv_lim(self):
        """get didv_lim"""
        return self.plot["didv_lim"]

    @didv_lim.setter
    def didv_lim(self, didv_lim: tuple):
        """set didv_lim"""
        self.plot["didv_lim"] = didv_lim
        logger.info("(%s) didv_lim = %s", self._name, didv_lim)

    @property
    def n_lim(self):
        """get n_lim"""
        return self.plot["n_lim"]

    @n_lim.setter
    def n_lim(self, n_lim: tuple):
        """set n_lim"""
        self.plot["n_lim"] = n_lim
        logger.info("(%s) n_lim = %s", self._name, n_lim)

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
    def z_cmap(self):
        """get z_cmap"""
        return self.plot["z_cmap"]

    @z_cmap.setter
    def z_cmap(self, z_cmap):
        """set z_cmap"""
        self.plot["z_cmap"] = z_cmap
        logger.info("(%s) z_cmap = %s", self._name, z_cmap)

    @property
    def z_contrast(self):
        """get z_contrast"""
        return self.plot["z_contrast"]

    @z_contrast.setter
    def z_contrast(self, z_contrast: float):
        """set z_contrast"""
        self.plot["z_contrast"] = z_contrast
        logger.info("(%s) z_contrast = %s", self._name, z_contrast)

    @property
    def n_show(self):
        """get n_show"""
        return self.plot["n_show"]

    @n_show.setter
    def n_show(self, n_show: bool):
        """set n_show"""
        self.plot["n_show"] = n_show
        logger.info("(%s) n_show = %s", self._name, n_show)

    @property
    def n_size(self):
        """get n_size"""
        return self.plot["n_size"]

    @n_size.setter
    def n_size(self, n_size: tuple[float, float]):
        """set n_size"""
        self.plot["n_size"] = n_size
        logger.info("(%s) n_size = %s", self._name, n_size)

    @property
    def n_labelsize(self):
        """get n_labelsize"""
        return self.plot["n_labelsize"]

    @n_labelsize.setter
    def n_labelsize(self, n_labelsize: float):
        """set n_labelsize"""
        self.plot["n_labelsize"] = n_labelsize
        logger.info("(%s) n_labelsize = %s", self._name, n_labelsize)

    @property
    def n_color(self):
        """get n_color"""
        return self.plot["n_color"]

    @n_color.setter
    def n_color(self, n_color):
        """set n_color"""
        self.plot["n_color"] = n_color
        logger.info("(%s) n_color = %s", self._name, n_color)

    @property
    def n_style(self):
        """get n_style"""
        return self.plot["n_style"]

    @n_style.setter
    def n_style(self, n_style: str):
        """set n_style"""
        self.plot["n_style"] = n_style
        logger.info("(%s) n_style = %s", self._name, n_style)

    @property
    def n_lwms(self):
        """get n_lwms"""
        return self.plot["n_lwms"]

    @n_lwms.setter
    def n_lwms(self, n_lwms: float):
        """set n_lwms"""
        self.plot["n_lwms"] = n_lwms
        logger.info("(%s) n_lwms = %s", self._name, n_lwms)

    @property
    def fig_size(self):
        """get fig_size"""
        return self.plot["fig_size"]

    @fig_size.setter
    def fig_size(self, fig_size: tuple[float]):
        """set fig_size"""
        self.plot["fig_size"] = fig_size
        logger.info("(%s) fig_size = %s", self._name, fig_size)

    @property
    def fig_nr(self):
        """get fig_nr"""
        return self.plot["fig_nr"]

    @fig_nr.setter
    def fig_nr(self, fig_nr: int):
        """set fig_nr"""
        self.plot["fig_nr"] = fig_nr
        logger.info("(%s) fig_nr = %s", self._name, fig_nr)

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
    def indices(self):
        """get indices"""
        return self.plot["indices"]

    @indices.setter
    def indices(self, indices: list[int]):
        """set indices"""
        self.plot["indices"] = indices
        logger.info("(%s) indices = %s", self._name, indices)

    @property
    def values(self):
        """get values"""
        return self.plot["values"]

    @values.setter
    def values(self, values: list[float]):
        """set values"""
        self.plot["values"] = values
        logger.info("(%s) values = %s", self._name, values)


'''


        self.show_all = {
            "indices": indices,
            "title": self.base["title"],
            "fig": fig,
            "ax_z": ax_z,
            "ax_c": ax_c,
            "ax_n": ax_n,
            "ax_i": ax_i,
            "ax_didv": ax_didv,
            "x_key": self.plot["x_key"],
            "y_key": self.plot["y_key"],
            "z_key": self.plot["z_key"],
            "v_key": self.plot["v_key"],
            "i_key": self.plot["i_key"],
            "didv_key": self.plot["didv_key"],
            "n_key": self.plot["n_key"],
            "x_lim": self.plot["x_lim"],
            "y_lim": self.plot["y_lim"],
            "z_lim": self.plot["z_lim"],
            "v_lim": self.plot["v_lim"],
            "i_lim": self.plot["i_lim"],
            "didv_lim": self.plot["didv_lim"],
            "n_lim": self.plot["n_lim"],
            "x_data": x_data,
            "y_data": y_data,
            "z_data": z_data,
            "v_data": v_data,
            "i_data": i_data,
            "didv_data": didv_data,
            "n_data": n_data,
            "x_label": ppk[self.plot["x_key"]][1],
            "y_label": ppk[self.plot["y_key"]][1],
            "z_label": ppk[self.plot["z_key"]][1],
            "v_label": ppk[self.plot["v_key"]][1],
            "i_label": ppk[self.plot["i_key"]][1],
            "didv_label": ppk[self.plot["didv_key"]][1],
            "n_label": ppk[self.plot["n_key"]][1],
            "fig_nr": self.plot["fig_nr"],
            "fig_size": self.plot["fig_size"],
            "display_dpi": self.plot["display_dpi"],
            "z_cmap": self.plot["z_cmap"],
            "z_contrast": self.plot["z_contrast"],
            "n_show": self.plot["n_show"],
            "n_color": self.plot["n_color"],
            "n_style": self.plot["n_style"],
            "n_lwms": self.plot["n_lwms"],
            "n_labelsize": self.plot[]
        }

    def reshowAll(
        self,
        fig_nr=None,
        title=None,
        sub_folder=None,
    ):
        """reshowIV()
        - reshows IV
        """
        logger.info(
            "(%s) reshowIV()",
            self._name,
        )

        if title is not None:
            self.title = title

        if sub_folder is not None:
            self.sub_folder = sub_folder

        with plt.ioff():
            self.loadData()

        if fig_nr is None:
            fig_nr = self.show_all["fig_nr"]

        plot_all(
            indices=self.show_all["indices"],
            x=self.show_all["x_data"],
            y=self.show_all["y_data"],
            z=self.show_all["z_data"],
            n=self.show_all["n_data"],
            i=self.show_all["i_data"],
            x_lim=self.show_all["x_lim"],
            x2_lim=self.show_all["x2_lim"],
            y_lim=self.show_all["y_lim"],
            z_lim=self.show_all["z_lim"],
            z2_lim=self.show_all["z2_lim"],
            n_lim=self.show_all["n_lim"],
            i_lim=self.show_all["i_lim"],
            x_label=self.show_all["x_label"],
            y_label=self.show_all["y_label"],
            z_label=self.show_all["z_label"],
            n_label=self.show_all["n_label"],
            i_label=self.show_all["i_label"],
            fig_nr=fig_nr,
            display_dpi=self.show_all["display_dpi"],
            n_color=self.show_all["n_color"],
            n_style=self.show_all["n_style"],
            n_lwms=self.show_all["n_lwms"],
            title=self.show_all["title"],
            n_show=self.show_all["n_show"],
        )
        plt.show()

    def saveAll(self, string=None):
        """saveIV()
        - safes Figure to self.figure_folder/self.title
        """
        logger.info(
            "(%s) saveIV() to %s%s.png",
            self._name,
            self.base["figure_folder"],
            self.base["title"],
        )
        if string is None:
            string = f"all {self.show_all['indices']}"
        self.saveFigure(figure=self.show_all["fig"], subtitle=string)

        '''
