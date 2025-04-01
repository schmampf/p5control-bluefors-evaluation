"""
BasePlot Module

This module provides the `BasePlot` class for handling and managing the saving and loading of matplotlib figures. It supports saving figures in PNG, PDF, and a pickled format (.fig.pickle) for later use.

Classes:
- BasePlot: A base class for saving and loading figures.

Dependencies:
- Python 3.10
- Matplotlib
- OS, Sys, Logging, Pickle

Usage:
    plot = BasePlot(name="MyPlot")
    fig, ax = plt.subplots()
    plot.saveFigure(fig, sub_title="example")

Author: Oliver Irtenkauf
Date: 2025-04-01
"""

import os
import sys
import logging
import importlib
import pickle

import matplotlib
import matplotlib.figure

from utilities.baseclass import BaseClass


# Reload the base class module to ensure changes are applied if modified
importlib.reload(sys.modules["utilities.baseclass"])

# Set up logging for debugging and tracking events
logger = logging.getLogger(__name__)


class BasePlot(BaseClass):
    """
    A base class for handling figure plotting and saving operations.

    This class extends `BaseClass` and provides methods to save and load
    figures using different formats (PNG, PDF, and a pickled FIG format).
    """

    def __init__(
        self,
        name: str = "base plot",
    ):
        """
        Initialize a BasePlot instance.

        Parameters:
        - name (str): The name of the plot instance (default: "base plot").
        """
        super().__init__()
        self._base_plot_name = name

        logger.info("(%s) ... BasePlot initialized.", self._base_plot_name)

    def saveFigure(
        self,
        figure: matplotlib.figure.Figure,
        sub_title: str = "",
        sub_folder: str = "",
        save_as_png: bool = True,
        save_as_fig: bool = True,
        save_as_pdf: bool = False,
    ):
        """
        Save a matplotlib figure to a specified directory.

        Parameters:
        - figure (matplotlib.figure.Figure): The figure to save.
        - sub_title (str): Optional subtitle for the filename.
        - sub_folder (str): Optional subdirectory within the save path.
        - save_as_png (bool): Whether to save the figure as a PNG file (default: True).
        - save_as_fig (bool): Whether to save the figure as a pickled `.fig.pickle` file (default: True).
        - save_as_pdf (bool): Whether to save the figure as a PDF file (default: False).

        The figure is saved to:
        `self.figure_folder/self.sub_folder/self.title/sub_folder/sub_title`
        """

        logger.info("(%s) saveFigure()", self._base_plot_name)

        # Construct the folder path for saving the figure
        folder = os.path.join(
            os.getcwd(),
            self.figure_folder,
            self.sub_folder,
            self.title,
        )

        # Append optional subfolder if provided
        if sub_folder:
            folder = os.path.join(folder, sub_folder)

        # Create the directory if it does not exist
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Construct the full file path
        if sub_title:
            name = os.path.join(folder, sub_title)
        else:
            name = folder

        # Save the figure in different formats based on user preferences
        if save_as_png:
            figure.savefig(f"{name}.png", dpi=600)

        if save_as_fig:
            with open(f"{name}.fig.pickle", "wb") as file:
                pickle.dump(figure, file)

        if save_as_pdf:
            figure.savefig(f"{name}.pdf", dpi=600)

    def loadFigure(
        self,
        total_path: str = "",
        sub_title: str = "",
        sub_folder: str = "",
    ):
        """
        Load a previously saved figure from a pickle file.

        Parameters:
        - total_path (str): Full path to the pickled figure file.
        - sub_title (str): Optional subtitle to reconstruct the filename.
        - sub_folder (str): Optional subdirectory within the load path.

        Returns:
        - figure (matplotlib.figure.Figure): The loaded figure object.

        If `total_path` is provided, it directly loads the figure from the given path.
        Otherwise, it constructs the path based on `self.base` settings.
        """

        logger.info("(%s) loadFigure()", self._base_plot_name)

        # Use the given total path if provided
        if total_path:
            name = total_path
        else:
            # Construct the folder path based on base settings
            folder = os.path.join(
                os.getcwd(),
                self.figure_folder,
                self.sub_folder,
                self.title,
            )
            # Append optional subfolder if provided
            if not sub_folder:
                folder = os.path.join(
                    folder,
                    sub_folder,
                )

            # Construct the full file path
            if not sub_title:
                name = os.path.join(folder, sub_title)
            else:
                name = folder

        # Load the pickled figure
        with open(f"{name}.fig.pickle", "rb") as file:
            figure = pickle.load(file)

        return figure
