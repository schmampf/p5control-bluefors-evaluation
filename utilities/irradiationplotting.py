"""
Description.

"""

from ast import Tuple
import os
import sys
import subprocess
import logging
import importlib

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from tqdm import tqdm
from utilities.baseplotting import BasePlotting
from utilities.plot_all import plot_all
from utilities.key_database import PLOT_KEYS


importlib.reload(sys.modules["utilities.baseevaluation"])
importlib.reload(sys.modules["utilities.corporate_design_colors_v4"])
importlib.reload(sys.modules["utilities.plot_all"])
importlib.reload(sys.modules["utilities.key_database"])

logger = logging.getLogger(__name__)


def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


class IrradiationPlotting(BasePlotting):
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
        # super(IrradiationEvaluation, self).__init__(name=name)
        # super(BasePlotting, self).__init__(name=name)

        self.video = {
            "save_as_mp4": True,
            "save_as_gif": True,
            "frame_rate": 30,
            "video_size": (1200, 800),
        }

        self.plot["x_key"] = "irradiation_V_bias_up_mV"
        self.plot["nu_key"] = "irradiation_nu"
        self.plot["v_ac_key"] = "irradiation_v_ac"
        self.plot["z_key"] = "irradiation_dIdV_up"
        self.plot["v_key"] = "irradiation_V_bias_up_mV"
        self.plot["i_key"] = "irradiation_I_up_nA_abs"
        self.plot["didv_key"] = "irradiation_dIdV_up"
        self.plot["n_key"] = "irradiation_T_up_K"

        self.plot["v_ac_lim"] = (None, None)
        self.plot["nu_lim"] = (None, None)

        self.show_all_amplitude_studies = {}
        self.show_all_frequency_studies = {}

        logger.info("(%s) ... IrradiationPlotting initialized.", self._name)

    def createVideo(self, name: str):
        folder = os.path.join(
            os.getcwd(),
            self.base["figure_folder"],
            self.base["sub_folder"],
            self.base["title"],
        )
        folder = str(folder)
        folder = folder.replace(" ", "\\ ")
        folder = folder.replace("(", "\\(")
        folder = folder.replace(")", "\\)")

        if self.save_as_mp4:
            bashCommand = (
                "ffmpeg"
                + " -y"
                + f" -r {self.frame_rate}"
                + f" -i {folder}{name}/%06d.png"
                + f' -vf scale="{self.video_size[0]}:{self.video_size[1]}"'
                + f" -r {self.frame_rate}"
                + f" {folder}{name}.mp4"
            )
            subprocess.call(
                bashCommand,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

        if self.save_as_gif:
            bashCommand = (
                "ffmpeg"
                + " -y"
                + f" -r {self.frame_rate}"
                + f" -i {folder}{name}/%06d.png"
                + f' -vf scale="{self.video_size[0]}:{self.video_size[1]}"'
                + f" -framerate {self.frame_rate}"
                + " -pix_fmt rgb24"
                + " -loop 0"
                + f" {folder}{name}.gif"
            )
            subprocess.call(
                bashCommand,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

    def createAmplitudeStudiesVideo(self):
        """createAmplitudeStudiesVideo()
        - calls createVideo for "/amplitude\\ studies"
        """
        logger.info(
            "(%s) createAmplitudeStudiesVideo()",
            self._name,
        )
        self.createVideo(name="/amplitude\\ studies")

    def createFrequencyStudiesVideo(self):
        """createFrequencyStudiesVideo()
        - calls createVideo for "/frequency\\ studies"
        """
        logger.info(
            "(%s) createFrequencyStudiesVideo()",
            self._name,
        )
        self.createVideo(name="/frequency\\ studies")

    def createIrradiationStudiesVideo(self):
        """createIrradiationStudiesVideo()
        - calls createVideo for "/irradiation\\ studies"
        """
        logger.info(
            "(%s) createIrradiationStudiesVideo()",
            self._name,
        )
        self.createVideo(name="/irradiation\\ studies")

    def saveAllAmplitudeStudies(
        self,
    ):
        """saveAllAmplitudeStudies()
        - safes Figure to self.figure_folder/self.title/sub_title
        """
        logger.info(
            "(%s) saveAllAmplitudeStudies() to %s%s%s",
            self._name,
            self.base["figure_folder"],
            self.base["title"],
            "/amplitude studies",
        )
        for i, fig in enumerate(tqdm(self.show_all_amplitude_studies["fig"])):
            self.saveFigure(
                figure=fig, subtitle=f"{i+1:06d}", subfolder="amplitude studies/"
            )

    def saveAllFrequencyStudies(
        self,
    ):
        """saveAllAmplitudeStudies()
        - safes Figure to self.figure_folder/self.title/sub_title
        """
        logger.info(
            "(%s) saveAllFrequencyStudies() to %s%s%s",
            self._name,
            self.base["figure_folder"],
            self.base["title"],
            "/frequency studies",
        )
        for i, fig in enumerate(tqdm(self.show_all_frequency_studies["fig"])):
            self.saveFigure(
                figure=fig, subtitle=f"{i+1:06d}", subfolder="frequency studies/"
            )

    def saveAllIrradiationStudies(
        self,
    ):
        """saveAllIrradiationStudies()
        - safes Figure to self.figure_folder/self.title/sub_title
        """
        logger.info(
            "(%s) saveAllIrradiationStudies() to %s%s%s",
            self._name,
            self.base["figure_folder"],
            self.base["title"],
            "/irradiation studies",
        )
        for i, fig in enumerate(tqdm(self.show_all_irradiation_studies["fig"])):
            self.saveFigure(
                figure=fig, subtitle=f"{i+1:06d}", subfolder="irradiation studies/"
            )

    def showAllAmplitudeStudies(
        self,
    ):
        """
        - calls plot_all()

        """
        logger.info(
            "(%s) showAllAmplitudeStudies()",
            self._name,
        )

        ppk = self.possible_plot_keys
        x_data = eval(ppk[self.x_key][0])
        nu_data = eval(ppk[self.nu_key][0])
        v_ac_data = eval(ppk[self.v_ac_key][0])
        z_data = eval(ppk[self.z_key][0])
        v_data = eval(ppk[self.v_key][0])
        i_data = eval(ppk[self.i_key][0])
        didv_data = eval(ppk[self.didv_key][0])
        n_data = eval(ppk[self.n_key][0])

        y_data = v_ac_data
        y_lim = self.v_ac_lim
        y_label = ppk[self.v_ac_key][1]

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

        indices = self.getIndices(y_data=v_ac_data)

        figs = []
        ax_zs = []
        ax_cs = []
        ax_ns = []
        ax_is = []
        ax_didvs = []
        sub_titles = []

        for i, nu in enumerate(tqdm(nu_data)):
            sub_title = rf"Amplitude Study ($\nu = {nu:05.2f}\,$GHz)"
            sub_titles.append(sub_title)

            fig, ax_z, ax_c, ax_n, ax_i, ax_didv = plot_all(
                indices=indices,
                x=x_data,
                y=y_data,
                z=z_data[i, :, :],
                v=v_data,
                i=i_data[i, :, :],
                didv=didv_data[i, :, :],
                n=n_data[i, :],
                x_lim=self.x_lim,
                y_lim=y_lim,
                z_lim=self.z_lim,
                v_lim=self.v_lim,
                i_lim=self.i_lim,
                didv_lim=self.didv_lim,
                n_lim=self.n_lim,
                x_label=ppk[self.x_key][1],
                y_label=y_label,
                z_label=ppk[self.z_key][1],
                v_label=ppk[self.v_key][1],
                i_label=ppk[self.i_key][1],
                didv_label=ppk[self.didv_key][1],
                n_label=ppk[self.n_key][1],
                fig_nr=self.fig_nr + i,
                fig_size=self.fig_size,
                display_dpi=self.display_dpi,
                z_cmap=self.z_cmap,
                z_contrast=self.z_contrast,
                n_show=self.n_show,
                n_size=self.n_size,
                n_labelsize=self.n_labelsize,
                n_color=self.n_color,
                n_style=self.n_style,
                n_lwms=self.n_lwms,
                title=sub_title,
            )
            figs.append(fig)
            ax_zs.append(ax_z)
            ax_cs.append(ax_c)
            ax_ns.append(ax_n)
            ax_is.append(ax_i)
            ax_didvs.append(ax_didv)

        self.show_all_amplitude_studies = {
            "title": self.title,
            "indices": indices,
            "sub_title": sub_titles,
            "fig": figs,
            "ax_z": ax_zs,
            "ax_c": ax_cs,
            "ax_n": ax_ns,
            "ax_i": ax_is,
            "ax_didv": ax_didvs,
        }

    def showAllFrequencyStudies(
        self,
    ):
        """
        - calls plot_all()

        """
        logger.info(
            "(%s) showAllFrequencyStudies()",
            self._name,
        )

        ppk = self.possible_plot_keys
        x_data = eval(ppk[self.x_key][0])
        nu_data = eval(ppk[self.nu_key][0])
        v_ac_data = eval(ppk[self.v_ac_key][0])
        z_data = eval(ppk[self.z_key][0])
        v_data = eval(ppk[self.v_key][0])
        i_data = eval(ppk[self.i_key][0])
        didv_data = eval(ppk[self.didv_key][0])
        n_data = eval(ppk[self.n_key][0])

        y_data = nu_data
        y_lim = self.nu_lim
        y_label = ppk[self.nu_key][1]

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

        indices = self.getIndices(y_data=nu_data)

        figs = []
        ax_zs = []
        ax_cs = []
        ax_ns = []
        ax_is = []
        ax_didvs = []
        sub_titles = []

        for i, v_ac in enumerate(tqdm(v_ac_data)):
            sub_title = rf"Frequency Study ($V = {v_ac:4.2f}\,$V)"
            sub_titles.append(sub_title)

            fig, ax_z, ax_c, ax_n, ax_i, ax_didv = plot_all(
                indices=indices,
                x=x_data,
                y=y_data,
                z=z_data[:, i, :],
                v=v_data,
                i=i_data[:, i, :],
                didv=didv_data[:, i, :],
                n=n_data[:, i],
                x_lim=self.x_lim,
                y_lim=y_lim,
                z_lim=self.z_lim,
                v_lim=self.v_lim,
                i_lim=self.i_lim,
                didv_lim=self.didv_lim,
                n_lim=self.n_lim,
                x_label=ppk[self.x_key][1],
                y_label=y_label,
                z_label=ppk[self.z_key][1],
                v_label=ppk[self.v_key][1],
                i_label=ppk[self.i_key][1],
                didv_label=ppk[self.didv_key][1],
                n_label=ppk[self.n_key][1],
                fig_nr=self.fig_nr + i,
                fig_size=self.fig_size,
                display_dpi=self.display_dpi,
                z_cmap=self.z_cmap,
                z_contrast=self.z_contrast,
                n_show=self.n_show,
                n_size=self.n_size,
                n_labelsize=self.n_labelsize,
                n_color=self.n_color,
                n_style=self.n_style,
                n_lwms=self.n_lwms,
                title=sub_title,
            )
            figs.append(fig)
            ax_zs.append(ax_z)
            ax_cs.append(ax_c)
            ax_ns.append(ax_n)
            ax_is.append(ax_i)
            ax_didvs.append(ax_didv)

        self.show_all_frequency_studies = {
            "title": self.title,
            "indices": indices,
            "sub_title": sub_titles,
            "fig": figs,
            "ax_z": ax_zs,
            "ax_c": ax_cs,
            "ax_n": ax_ns,
            "ax_i": ax_is,
            "ax_didv": ax_didvs,
        }

    def showAllIrradiationStudies(
        self,
    ):
        """
        - calls plot_all()

        """
        logger.info(
            "(%s) showAllFrequencyStudies()",
            self._name,
        )

        ppk = self.possible_plot_keys
        V_data = eval(ppk[self.x_key][0])
        nu_data = eval(ppk[self.nu_key][0])
        v_ac_data = eval(ppk[self.v_ac_key][0])
        z_data = eval(ppk[self.z_key][0])
        # v_data = eval(ppk[self.v_key][0])
        i_data = np.abs(eval(ppk[self.i_key][0]))
        didv_data = eval(ppk[self.didv_key][0])
        n_data = eval(ppk[self.n_key][0])

        x_data = v_ac_data
        x_lim = self.v_ac_lim
        x_label = ppk[self.v_ac_key][1]

        y_data = nu_data
        y_lim = self.nu_lim
        y_label = ppk[self.nu_key][1]

        v_data = x_data
        v_lim = x_lim[1]
        v_label = x_label

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

        indices = self.getIndices(y_data=nu_data)

        figs = []
        ax_zs = []
        ax_cs = []
        ax_ns = []
        ax_is = []
        ax_didvs = []
        sub_titles = []

        for i, V in enumerate(tqdm(V_data)):
            sub_title = rf"Irradiation Study ($V = {V:6.4f}\,$V)"
            sub_titles.append(sub_title)

            fig, ax_z, ax_c, ax_n, ax_i, ax_didv = plot_all(
                indices=indices,
                x=x_data,
                y=y_data,
                z=z_data[:, :, i],
                v=v_data,
                i=i_data[:, :, i],
                didv=didv_data[:, :, i],
                n=n_data,
                x_lim=x_lim,
                y_lim=y_lim,
                z_lim=self.z_lim,
                v_lim=v_lim,
                i_lim=self.i_lim,
                didv_lim=self.didv_lim,
                n_lim=self.n_lim,
                x_label=x_label,
                y_label=y_label,
                z_label=ppk[self.z_key][1],
                v_label=v_label,
                i_label=ppk[self.i_key][1],
                didv_label=ppk[self.didv_key][1],
                n_label=ppk[self.n_key][1],
                fig_nr=self.fig_nr + i,
                fig_size=self.fig_size,
                display_dpi=self.display_dpi,
                z_cmap=self.z_cmap,
                z_contrast=self.z_contrast,
                n_show=False,
                n_size=self.n_size,
                n_labelsize=self.n_labelsize,
                n_color=self.n_color,
                n_style=self.n_style,
                n_lwms=self.n_lwms,
                title=sub_title,
            )
            figs.append(fig)
            ax_zs.append(ax_z)
            ax_cs.append(ax_c)
            ax_ns.append(ax_n)
            ax_is.append(ax_i)
            ax_didvs.append(ax_didv)

        self.show_all_irradiation_studies = {
            "title": self.title,
            "indices": indices,
            "sub_title": sub_titles,
            "fig": figs,
            "ax_z": ax_zs,
            "ax_c": ax_cs,
            "ax_n": ax_ns,
            "ax_i": ax_is,
            "ax_didv": ax_didvs,
        }

    @property
    def nu_key(self):
        """get nu_key"""
        return self.plot["nu_key"]

    @nu_key.setter
    def nu_key(self, nu_key: str):
        """set nu_key"""
        self.plot["nu_key"] = nu_key
        logger.info("(%s) nu_key = %s", self._name, nu_key)

    @property
    def v_ac_key(self):
        """get v_ac_key"""
        return self.plot["v_ac_key"]

    @v_ac_key.setter
    def v_ac_key(self, v_ac_key: str):
        """set v_ac_key"""
        self.plot["v_ac_key"] = v_ac_key
        logger.info("(%s) v_ac_key = %s", self._name, v_ac_key)

    @property
    def v_ac_lim(self):
        """get v_ac_lim"""
        return self.plot["v_ac_lim"]

    @v_ac_lim.setter
    def v_ac_lim(self, v_ac_lim: tuple):
        """set v_ac_lim"""
        self.plot["v_ac_lim"] = v_ac_lim
        logger.info("(%s) v_ac_lim = %s", self._name, v_ac_lim)

    @property
    def nu_lim(self):
        """get nu_lim"""
        return self.plot["nu_lim"]

    @nu_lim.setter
    def nu_lim(self, nu_lim: tuple):
        """set nu_lim"""
        self.plot["nu_lim"] = nu_lim
        logger.info("(%s) nu_lim = %s", self._name, nu_lim)

    @property
    def save_as_mp4(self):
        """get save_as_mp4"""
        return self.video["save_as_mp4"]

    @save_as_mp4.setter
    def save_as_mp4(self, save_as_mp4: bool):
        """set save_as_mp4"""
        self.video["save_as_mp4"] = save_as_mp4
        logger.info("(%s) save_as_mp4 = %s", self._name, save_as_mp4)

    @property
    def save_as_gif(self):
        """get save_as_gif"""
        return self.video["save_as_gif"]

    @save_as_gif.setter
    def save_as_gif(self, save_as_gif: bool):
        """set save_as_gif"""
        self.video["save_as_gif"] = save_as_gif
        logger.info("(%s) save_as_gif = %s", self._name, save_as_gif)

    @property
    def video_size(self):
        """get video_size"""
        return self.video["video_size"]

    @video_size.setter
    def video_size(self, video_size: tuple):
        """set video_size"""
        self.video["video_size"] = video_size
        logger.info("(%s) video_size = %s", self._name, video_size)

    @property
    def frame_rate(self):
        """get frame_rate"""
        return self.video["frame_rate"]

    @frame_rate.setter
    def frame_rate(self, frame_rate: int):
        """set frame_rate"""
        self.video["frame_rate"] = frame_rate
        logger.info("(%s) frame_rate = %s", self._name, frame_rate)
