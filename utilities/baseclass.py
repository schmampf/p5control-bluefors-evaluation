"""Module of a base class that can save/load/show its own dictionary."""

import os
import sys
import pickle
import logging
import platform
from importlib import reload

from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QIcon

from utilities.hdf5view.mainwindow import MainWindow

reload(logging)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class BaseClass:
    """
    Base Class has method to save/load/show available data
    """

    def __init__(
        self,
        name="eva",
    ):
        """
        Description
        """
        # initialize name
        self._name = name

        # initialize logger
        match platform.system():
            case "Darwin":
                file_directory = "/Users/oliver/Documents/measurement_data/"
            case "Linux":
                file_directory = "/home/oliver/Documents/measurement_data/"
            case default:
                file_directory = ""
                logger.warning(
                    "(%s) needs a file directory under %s.",
                    self._name,
                    default,
                )

        # where to save fiure/data and find the measurement file
        self.base = {
            "title": "",
            "sub_folder": "",
            "data_folder": "data/",
            "figure_folder": "figures/",
            "ignore_while_saving": ["_name"],
            "file_directory": file_directory,
            "file_folder": "",
            "file_name": "",
        }
        logger.info("(%s) ... BaseClass initialized.", self._name)

    def viewFile(self):
        """
        Opens hdf5view GUI to look into hdf5 file.
        """
        logger.info("(%s) viewFile()", self._name)

        file_name = f"{self.base['file_directory']}{self.base['file_folder']}{self.base['file_name']}"
        app = QApplication(sys.argv)
        app.setOrganizationName("hdf5view")
        app.setApplicationName("hdf5view")
        app.setWindowIcon(QIcon("icons:hdf5view.svg"))
        window = MainWindow(app)
        window.show()
        window.open_file(file_name)
        app.exec_()

        # file_name = file_name.replace(" ", "\ ")
        # command = f"hdf5view -f {file_name}"
        # os.system(command)

    def saveData(
        self,
        title=None,
    ):
        """saveData()
        - safes self.__dict__ to pickle
        """
        logger.info("(%s) saveData()", self._name)

        # Handle Title
        if title is None:
            title = f"{self.base['title']}.pickle"

        # Handle data folder
        folder = os.path.join(
            os.getcwd(), self.base["data_folder"], self.base["sub_folder"]
        )
        check = os.path.isdir(folder)
        if not check and self.base["data_folder"] != "":
            os.makedirs(folder)

        # Get Dictionary
        data = {}
        for key, value in self.__dict__.items():
            if key not in self.base["ignore_while_saving"]:
                data[key] = value

        # save data to pickle
        name = os.path.join(
            os.getcwd(), self.base["data_folder"], self.base["sub_folder"], title
        )
        with open(name, "wb") as file:
            pickle.dump(data, file)

    def loadData(
        self,
        title=None,
    ):
        """loadData()
        - loads self.__dict__ from pickle
        """
        logger.info("(%s) loadData()", self._name)

        # Handle Title
        if title is None:
            title = f"{self.base['title']}.pickle"

        # get data from pickle
        name = os.path.join(
            os.getcwd(), self.base["data_folder"], self.base["sub_folder"], title
        )
        with open(name, "rb") as file:
            data = pickle.load(file)

        # Save Data to self.
        for key, value in data.items():
            self.__dict__[key] = value

    def showData(
        self,
    ):
        """showData()
        - shows self.__dict__
        """
        logger.info("(%s) showData()", self._name)

        # Get Dictionary
        data = {}
        for key, value in self.__dict__.items():
            if key not in self.base["ignore_while_saving"]:
                data[key] = value
        return data

    @property
    def title(self):
        """get title"""
        return self.base["title"]

    @title.setter
    def title(self, title: str):
        """set title"""
        self.base["title"] = title
        logger.info("(%s) title = %s", self._name, title)

    @property
    def sub_folder(self):
        """get sub_folder"""
        return self.base["sub_folder"]

    @sub_folder.setter
    def sub_folder(self, sub_folder: str):
        """set sub_folder"""
        self.base["sub_folder"] = sub_folder
        logger.info("(%s) sub_folder = %s", self._name, sub_folder)

    @property
    def data_folder(self):
        """get data_folder"""
        return self.base["data_folder"]

    @data_folder.setter
    def data_folder(self, data_folder: str):
        """set data_folder"""
        self.base["data_folder"] = data_folder
        logger.info("(%s) data_folder = %s", self._name, data_folder)

    @property
    def figure_folder(self):
        """get figure_folder"""
        return self.base["figure_folder"]

    @figure_folder.setter
    def figure_folder(self, figure_folder: str):
        """set figure_folder"""
        self.base["figure_folder"] = figure_folder
        logger.info("(%s) figure_folder = %s", self._name, figure_folder)

    @property
    def ignore_while_saving(self):
        """get ignore_while_saving"""
        return self.base["ignore_while_saving"]

    @ignore_while_saving.setter
    def ignore_while_saving(self, ignore_while_saving: str):
        """set ignore_while_saving"""
        self.base["ignore_while_saving"] = ignore_while_saving
        logger.info("(%s) ignore_while_saving = %s", self._name, ignore_while_saving)

    @property
    def file_directory(self):
        """get file_directory"""
        return self.base["file_directory"]

    @file_directory.setter
    def file_directory(self, file_directory: str):
        """set file_directory"""
        self.base["file_directory"] = file_directory
        logger.info("(%s) file_directory = %s", self._name, file_directory)

    @property
    def file_folder(self):
        """get file_folder"""
        return self.base["file_folder"]

    @file_folder.setter
    def file_folder(self, file_folder: str):
        """set file_folder"""
        self.base["file_folder"] = file_folder
        logger.info("(%s) file_folder = %s", self._name, file_folder)

    @property
    def file_name(self):
        """get file_name"""
        return self.base["file_name"]

    @file_name.setter
    def file_name(self, file_name: str):
        """set file_name"""
        self.base["file_name"] = file_name
        logger.info("(%s) file_name = %s", self._name, file_name)
