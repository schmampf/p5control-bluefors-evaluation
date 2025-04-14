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

Author: Oliver Irtenkauf, David Capalbo
Creation: 2025-04-01
Last Modified: 2025-04-14
"""

# region imports 
# std lib
import os
import sys
import platform
# from enum import Enum, auto
# from typing import Dict, Any
from dataclasses import dataclass, field
from typing import Dict, Any, List
# from abc import ABC, abstractmethod

# 3rd party
import pickle
# import logging
# from importlib import reload
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon

# local
from utilities.hdf5view.mainwindow import MainWindow
from utilities.logger import logger
# endregion

@dataclass
class FileData:
    """FileData class to hold file-related attributes."""
    name: str                   = field(default="default")
    title: str                  = field(default="")
    sub_folder: str             = field(default="")
    data_folder: str            = field(default="data/")
    figure_folder: str          = field(default="figures/")
    file_directory: str         = field(default="")
    file_folder: str            = field(default="")
    file_name: str              = field(default="")

@dataclass
class DataCollection:
    packets: Dict[str, Any]             = field(default_factory=lambda: {"data": FileData()})
    fields_ignore_on_save: List[str]    = field(default_factory=lambda: ["name", "plot_name"])
   
class FileAPI:
    """
    FileAPI provides methods to save and load data files.
    It also integrates HDF5 file visualization through a Qt-based GUI.
    """
    
    @staticmethod
    def saveData(collection: DataCollection, title: str = ""):
        """
        Save the object's attributes to a file.
        """
        
        logger.info("(%s) saveData()", collection.packets["data"].name)
        if not title:
            title = f"{collection.packets["data"].title}.pickle"

        # Ensure data folder exists
        folder = os.path.join(os.getcwd(), collection.packets["data"].data_folder, collection.packets["data"].sub_folder)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Filter data for saving
        to_store = []
        for label, packet in collection.packets.items():    # for each data group/packet
            for key, value in packet.__dict__.items():      # for each attribute in the packet
                if key not in collection.fields_ignore_on_save: # ignore specified attributes
                    to_store.append((label, key, value))   # store the attribute if not ignored
        
        # Save data
        filepath = os.path.join(folder, title)
        with open(filepath, "wb") as file:
            pickle.dump(to_store, file)
    
    @staticmethod
    def loadData(collection: DataCollection, title: str = ""):
        """
        Load attributes from a previously saved file.
        """
        
        logger.info("(%s) loadData()", collection.packets["data"].name)
        if not title:
            title = f"{collection.packets["data"].title}.pickle"

        filepath = os.path.join(os.getcwd(), collection.packets["data"].data_folder, collection.packets["data"].sub_folder, title)
        with open(filepath, "rb") as file:
            loaded_data = pickle.load(file)
            
        # for packet in collection.packets.values():
        #     for key, value in packet.__dict__.items():
        #         print(f"key: {key}, value: {value}")   
                # if key in collection.fields_ignore_on_save:
                #     del packet.__dict__[key]

        # Update the data object with loaded attributes
        for label, key, value in loaded_data:
            if key not in collection.fields_ignore_on_save:
                collection.packets[label].__dict__[key] = value
        
        print(collection.packets["data"].__dict__)
    
    @staticmethod
    def showData(collection: DataCollection):
        """Display stored attributes excluding those marked for ignoring."""
        
        logger.info("(%s) showData()", collection.packets["data"].name)
        to_show = {}
        for label, packet in collection.packets.items():
            for key, value in packet.__dict__.items():
                if key not in collection.fields_ignore_on_save:
                    to_show[f"{label}.{key}"] = value
        return to_show
        
    @staticmethod
    def showFile(data: FileData):
        """Display the HDF5 file using the GUI."""
        logger.info("(%s) showFile()", data.name)
        file_name = os.path.join(
            data.file_directory,
            data.file_folder,
            data.file_name,
        )

        app = QApplication(sys.argv)
        app.setOrganizationName("hdf5view")
        app.setApplicationName("hdf5view")
        app.setWindowIcon(QIcon("icons:hdf5view.svg"))
        window = MainWindow(app)
        window.show()
        window.open_file(file_name)
        app.exec()

# class Files(FileData, FileAPI):
#     data: FileData
    
#     def __setattr__(self, name, value):
#         if not name == "data":  
#             logger.debug("(%s) (%s) = %s", self.data.name, name, value)
#             self.data.__setattr__(name, value)
#         else:
#             super().__setattr__(name, value)
    
#     def __getattr__(self, name):
#         if not name == "data":
#             logger.debug("(%s) (%s)", self.data.name, name)
#             return self.data.__getattribute__(name)
#         else:
#             return super().__getattribute__(name)
    
#     def __init__(self, name="base", root_dir=None):
#         self.data = FileData(name=name)
        
#         if not root_dir:
#             logger.warning("(%s) The project root directory not set. Assuming defaults.", self.data.name)
            
#             username = os.getlogin()
            
#             match (platform.system()):
#                 case "Darwin":
#                     file_directory = f"/Volumes/{username}/measurement data/"
#                 case "Linux":
#                     file_directory = f"/home/{username}/Documents/measurement_data/"
#         else:
#             if not root_dir.endswith("/"):
#                 root_dir += "/"
            
#             file_directory = root_dir
#             logger.info("(%s) The project root directory set to %s.", self.data.name, file_directory)
        
#         self.data.file_directory = file_directory
#         logger.info("(%s) ... Files initialized.", self.data.name)     

#     def saveData(self, title: str = ""):
#         """Save the object's attributes to a file."""
#         collection = DataCollection()
#         collection.packets["data"] = self.data
#         super().saveData(collection, title)
        
#     def loadData(self, title: str = ""):
#         """Load attributes from a previously saved file."""
#         collection = DataCollection()
#         collection.packets["data"] = self.data  # set packet state to current (keep ignored value state)
#         super().loadData(collection, title)
#         self.data = collection.packets["data"]  # update the data object with loaded attributes
        
#     def showData(self):
#         """Display stored attributes excluding those marked for ignoring."""
#         collection = DataCollection()
#         collection.packets["data"] = self.data
#         return super().showData(collection)
    
#     def showFile(self):
#         """Display the HDF5 file using the GUI."""
#         super().showFile(self.data)
        
# def test():
#     """Test the Files class."""
#     obj = Files(name="MyBase")
#     obj.saveData("test_data.pickle")
#     obj.loadData("test_data.pickle")
#     print(obj.showData())
    
#     obj.title = "MyTitle"
#     print(obj.title)
# # test()