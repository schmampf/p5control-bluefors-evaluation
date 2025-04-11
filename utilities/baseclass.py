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

# region imports 
# std lib
import os
import sys
import platform
from enum import Enum, auto
from typing import Dict, Any

# 3rd party
import pickle
import logging
from importlib import reload
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon

# local
from utilities.hdf5view.mainwindow import MainWindow
# endregion

# region Configure logging
reload(logging)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
# endregion

class BaseFields(Enum):
    """A collection of valid BaseFields for the BaseClass."""
    title               = auto()
    sub_folder          = auto()
    data_folder         = auto()
    figure_folder       = auto()
    ignore_while_saving = auto()
    file_directory      = auto()
    file_folder         = auto()
    file_name           = auto()

class BaseClass:
    """
    BaseClass provides methods to save, load, and display its attributes.
    It also integrates HDF5 file visualization through a Qt-based GUI.
    """
    
    name: str
    base: Dict[BaseFields, Any] = {}
    
    # region overload setter and getters
    def __setattr__(self, name, value):
        key = BaseFields[name] if name in BaseFields._member_names_ else None
        if key:
            self.base[key] = value
        else:
            super().__setattr__(name, value)
    
    def __getattr__(self, name):
        key = BaseFields[name] if name in BaseFields._member_names_ else None
        if key:
            return self.base[key]
        else:
            return super().__getattribute__(name)
    # endregion
    
    def __init__(self, 
        name="base",
        root_dir=None,
    ):
        """
        Initialize the BaseClass instance with default storage settings.

        Args:
            name (str): The name of the instance.
        """
        self.name = name
        
        if not root_dir:
            logger.warning("(%s) The project root directory not set. Assuming defaults.", self.name)
            
            username = os.getlogin()
            
            match (platform.system()):
                case "Darwin":
                    file_directory = f"/Volumes/{username}/measurement data/"
                case "Linux":
                    file_directory = f"/home/{username}/Documents/measurement_data/"
        else:
            if not root_dir.endswith("/"):
                root_dir += "/"
            
            file_directory = root_dir
            logger.info("(%s) The project root directory set to %s.", self.name, file_directory)                

        # Define base configuration for file management
        self.base = {
            BaseFields.title:               "",
            BaseFields.sub_folder:          "",
            BaseFields.data_folder:         "data/",
            BaseFields.figure_folder:       "figures/",
            BaseFields.ignore_while_saving: ["name", "_base_plot_name"],
            BaseFields.file_directory:      file_directory,
            BaseFields.file_folder:         "",
            BaseFields.file_name:           "",
        }
        logger.info("(%s) ... BaseClass initialized.", self.name)

    def showFile(self):
        """
        Open the HDF5 file using a Qt-based GUI viewer.
        """
        logger.info("(%s) showFile()", self.name)
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
        app.exec()

    def saveData(self, title: str = ""):
        """
        Save the object's attributes to a pickle file.

        Args:
            title (str, optional): The filename for the saved data. Defaults to base title.
        """
        logger.info("(%s) saveData()", self.name)
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
        logger.info("(%s) loadData()", self.name)
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
        logger.info("(%s) showData()", self.name)
        return {
            k: v for k, v in self.__dict__.items() if k not in self.ignore_while_saving
        }