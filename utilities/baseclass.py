"""
BaseClass Module

This module provides a base class for managing data storage and retrieval.
It allows saving, loading, and displaying its attributes as a dictionary.

Features:
- Save and load attributes using pickle.
- Open an HDF5 file using a GUI.
- Manage figure and data storage directories.
- Logging support for debugging and tracking operations.

Dependencies:
- Python 3.10
- OS, Sys, Logging, Pickle, Platform
- Matplotlib
- qtpy for GUI interactions

Usage Example:
    obj = BaseClass(name="MyBase")
    obj.saveData("test_data.pickle")
    obj.loadData("test_data.pickle")
    print(obj.showData())

Author: Oliver Irtenkauf
Date: 2025-04-01
"""

import os
import sys
import pickle
import logging
import platform
from importlib import reload
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon
from hdf5view.mainwindow import MainWindow

# Configure logging
reload(logging)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class BaseClass:
    """
    BaseClass provides methods to save, load, and display its attributes.
    It also integrates HDF5 file visualization through a Qt-based GUI.
    """

    def __init__(self, name="base"):
        """
        Initialize the BaseClass instance with default storage settings.

        Args:
            name (str): The name of the instance.
        """
        self._base_name = name

        # Determine file directory based on OS
        if platform.system() == "Darwin":
            file_directory = "/Volumes/speedyboy/measurement data/"
        elif platform.system() == "Linux":
            file_directory = "/home/oliver/Documents/measurement_data/"
        else:
            file_directory = ""
            logger.warning("(%s) needs a valid file directory.", self._base_name)

        # Define base configuration for file management
        self.base = {
            "title": "",
            "sub_folder": "",
            "data_folder": "data/",
            "figure_folder": "figures/",
            "ignore_while_saving": ["_base_name", "_base_plot_name"],
            "file_directory": file_directory,
            "file_folder": "",
            "file_name": "",
        }
        logger.info("(%s) ... BaseClass initialized.", self._base_name)

    def showFile(self):
        """
        Open the HDF5 file using a Qt-based GUI viewer.
        """
        logger.info("(%s) showFile()", self._base_name)
        file_name = os.path.join(
            self.file_directory,
            self.file_folder,
            self.file_name,
        )

        app = QApplication(sys.argv)
        app.setOrganizationName("hdf5view")
        app.setApplicationName("hdf5view")
        app.setWindowIcon(QIcon("icons:hdf5view.svg"))
        window = MainWindow(app)
        window.show()
        window.open_file(file_name)
        app.exec_()

    def saveData(self, title: str = ""):
        """
        Save the object's attributes to a pickle file.

        Args:
            title (str, optional): The filename for the saved data. Defaults to base title.
        """
        logger.info("(%s) saveData()", self._base_name)
        if not title:
            title = f"{self.title}.pickle"

        # Ensure data folder exists
        folder = os.path.join(os.getcwd(), self.data_folder, self.sub_folder)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Filter data for saving
        data = {
            k: v for k, v in self.__dict__.items() if k not in self.ignore_while_saving
        }

        # Save data
        filepath = os.path.join(folder, title)
        with open(filepath, "wb") as file:
            pickle.dump(data, file)

    def loadData(self, title: str = ""):
        """
        Load attributes from a previously saved pickle file.

        Args:
            title (str, optional): The filename to load. Defaults to base title.
        """
        logger.info("(%s) loadData()", self._base_name)
        if not title:
            title = f"{self.title}.pickle"

        filepath = os.path.join(os.getcwd(), self.data_folder, self.sub_folder, title)
        with open(filepath, "rb") as file:
            data = pickle.load(file)

        self.__dict__.update(data)

    def showData(self):
        """
        Display stored attributes excluding those marked for ignoring.

        Returns:
            dict: The stored attributes.
        """
        logger.info("(%s) showData()", self._base_name)
        return {
            k: v for k, v in self.__dict__.items() if k not in self.ignore_while_saving
        }

    @property
    def title(self):
        """get title"""
        return self.base["title"]

    @title.setter
    def title(self, title: str):
        """set title"""
        self.base["title"] = title
        logger.info("%s", title)

    @property
    def sub_folder(self):
        """get sub_folder"""
        return self.base["sub_folder"]

    @sub_folder.setter
    def sub_folder(self, sub_folder: str):
        """set sub_folder"""
        self.base["sub_folder"] = sub_folder
        logger.debug("(%s) sub_folder = %s", self._base_name, sub_folder)

    @property
    def data_folder(self):
        """get data_folder"""
        return self.base["data_folder"]

    @data_folder.setter
    def data_folder(self, data_folder: str):
        """set data_folder"""
        self.base["data_folder"] = data_folder
        logger.debug("(%s) data_folder = %s", self._base_name, data_folder)

    @property
    def figure_folder(self):
        """get figure_folder"""
        return self.base["figure_folder"]

    @figure_folder.setter
    def figure_folder(self, figure_folder: str):
        """set figure_folder"""
        self.base["figure_folder"] = figure_folder
        logger.debug("(%s) figure_folder = %s", self._base_name, figure_folder)

    @property
    def ignore_while_saving(self):
        """get ignore_while_saving"""
        return self.base["ignore_while_saving"]

    @ignore_while_saving.setter
    def ignore_while_saving(self, ignore_while_saving: str):
        """set ignore_while_saving"""
        self.base["ignore_while_saving"] = ignore_while_saving
        logger.debug(
            "(%s) ignore_while_saving = %s", self._base_name, ignore_while_saving
        )

    @property
    def file_directory(self):
        """get file_directory"""
        return self.base["file_directory"]

    @file_directory.setter
    def file_directory(self, file_directory: str):
        """set file_directory"""
        self.base["file_directory"] = file_directory
        logger.debug("(%s) file_directory = %s", self._base_name, file_directory)

    @property
    def file_folder(self):
        """get file_folder"""
        return self.base["file_folder"]

    @file_folder.setter
    def file_folder(self, file_folder: str):
        """set file_folder"""
        self.base["file_folder"] = file_folder
        logger.debug("(%s) file_folder = %s", self._base_name, file_folder)

    @property
    def file_name(self):
        """get file_name"""
        return self.base["file_name"]

    @file_name.setter
    def file_name(self, file_name: str):
        """set file_name"""
        self.base["file_name"] = file_name
        logger.debug("(%s) file_name = %s", self._base_name, file_name)
