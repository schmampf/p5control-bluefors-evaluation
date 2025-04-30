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
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

# 3rd party
import pickle
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon
import h5py

# local
from hdf5view.mainwindow import MainWindow
import utilities.logging as Logger

# endregion


@dataclass
class FileData:
    """FileData class to hold file-related attributes."""

    name: str = field(default="default")
    title: str = field(default="")
    sub_folder: str = field(default="")
    data_folder: str = field(default="data/")
    figure_folder: str = field(default="figures/")
    file_directory: str = field(default="")
    file_folder: str = field(default="")
    file_name: str = field(default="")

    def __setattr__(self, name: str, value: Any):
        if not self.name == "default":
            Logger.print(Logger.DEBUG, msg=f"FileData.{name} = {value}")
        object.__setattr__(self, name, value)


@dataclass
class DataCollection:
    packets: Dict[str, Any] = field(default_factory=lambda: {"data": FileData()})
    fields_ignore_on_save: List[str] = field(
        default_factory=lambda: ["name", "plot_name"]
    )

    def __getattribute__(self, name: str) -> Any:
        if name in ["packets", "fields_ignore_on_save"]:
            return object.__getattribute__(self, name)
        if name in self.packets.keys():
            return self.packets[name]
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "packets":
            object.__setattr__(self, name, value)
        elif name in self.packets.keys():
            self.packets[name] = value
        else:
            object.__setattr__(self, name, value)


def saveData(collection: DataCollection, title: str = ""):
    """
    Save the object's attributes to a file.
    """

    Logger.print(Logger.INFO, Logger.START, f"Files.saveData(title={title})")
    if not title:
        title = f"{collection.data.title}.pickle"

    # Ensure data folder exists
    folder = os.path.join(
        os.getcwd(),
        collection.data.data_folder,
        collection.data.sub_folder,
    )
    if not os.path.isdir(folder):
        os.makedirs(folder)

    # Filter data for saving
    to_store = []
    for label, packet in collection.packets.items():  # for each data group/packet
        for (
            key,
            value,
        ) in packet.__dict__.items():  # for each attribute in the packet
            if (
                key not in collection.fields_ignore_on_save
            ):  # ignore specified attributes
                to_store.append(
                    (label, key, value)
                )  # store the attribute if not ignored

    # Save data
    filepath = os.path.join(folder, title)
    Logger.print(Logger.DEBUG, msg=f"Location: {filepath}")
    with open(filepath, "wb") as file:
        pickle.dump(to_store, file)


def loadData(collection: DataCollection, title: str = ""):
    """
    Load attributes from a previously saved file.
    """

    Logger.print(Logger.INFO, Logger.START, f"Files.loadData(title={title})")

    if not title:
        title = f"{collection.data.title}.pickle"

    filepath = os.path.join(
        os.getcwd(),
        collection.data.data_folder,
        collection.data.sub_folder,
        title,
    )

    # Check if the file exists
    if not os.path.isfile(filepath):
        Logger.print(Logger.ERROR, msg=f"File not found: {filepath}")
        return

    with open(filepath, "rb") as file:
        loaded_data = pickle.load(file)

    # Update the data object with loaded attributes
    for label, key, value in loaded_data:
        if key not in collection.fields_ignore_on_save:
            collection.packets[label].__dict__[key] = value


def showData(collection: DataCollection):
    """Display stored attributes excluding those marked for ignoring."""

    Logger.print(Logger.INFO, Logger.START, "Files.showData()")

    to_show = {}
    for label, packet in collection.packets.items():
        for key, value in packet.__dict__.items():
            if key not in collection.fields_ignore_on_save:
                to_show[f"{label}.{key}"] = value

    for title, content in to_show.items():
        Logger.print(Logger.INFO, msg=f"{title}: {content}")


def showFile(data: FileData):
    """Display the HDF5 file using the GUI."""
    Logger.print(Logger.INFO, Logger.START, "Files.showDile()")
    file_name = getfile_path(data)

    app = QApplication(sys.argv)
    app.setOrganizationName("hdf5view")
    app.setApplicationName("hdf5view")
    app.setWindowIcon(QIcon("icons:hdf5view.svg"))
    window = MainWindow(app)
    window.show()
    window.open_file(file_name)
    app.exec()


def getfile_path(data: FileData) -> str:
    """Construct the full file path from the data attributes."""

    Logger.print(Logger.DEBUG, msg=f"Files.getfile_path()")

    return os.path.join(
        data.file_directory,
        data.file_folder,
        data.file_name,
    )


def open_file_group(
    data: FileData, dir: str
) -> Tuple[h5py.File | None, h5py.Group | None]:
    Logger.print(
        Logger.DEBUG,
        msg=f"Files.open_file(data, dir={dir})",
    )

    # check if file exists
    file_name = getfile_path(data)
    if not os.path.exists(file_name):
        Logger.print(Logger.ERROR, msg=f"Error: File does not exist: {file_name}")
        return (None, None)

    # show selected file
    Logger.print(Logger.DEBUG, msg=f"Opening file: {file_name}")

    file = h5py.File(file_name, "r")

    group = file.get(dir)
    if isinstance(group, h5py.Group):
        return file, group
    else:
        Logger.print(Logger.ERROR, msg=f"Error: Group not found: {dir}")
        return (None, None)


def ensure_group(obj):
    if isinstance(obj, h5py.Group):
        return obj
    raise TypeError(f"Expected an h5py.Group, but got {type(obj)}")


def ensure_file(obj):
    if isinstance(obj, h5py.File):
        return obj
    raise TypeError(f"Expected an h5py.File, but got {type(obj)}")


def setup(collection: DataCollection, name: str = "", root_dir: str = ""):
    """
    Setup the file data attributes and directory structure.
    If no root directory is provided, default directories are assumed.
    """

    Logger.print(
        Logger.INFO, Logger.START, f"Files.setup(name={name}, root_dir={root_dir})"
    )

    data = collection.packets["data"]
    data.name = name
    if not root_dir:
        Logger.print(
            Logger.WARNING, msg="The project root directory not set. Assuming defaults."
        )

        username = os.getlogin()

        match (platform.system()):
            case "Darwin":
                file_directory = f"/Volumes/{username}/measurement data/"
            case "Linux":
                file_directory = f"/home/{username}/Documents/measurement_data/"
    else:
        if not root_dir.endswith("/") and not root_dir.endswith("\\"):
            match (platform.system()):
                case "Windows":
                    root_dir += "\\"
                case "Linux":
                    root_dir += "/"

        file_directory = root_dir
    data.file_directory = file_directory
