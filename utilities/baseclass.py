"""Module of a base class that can save/load/show its own dictionary."""

import os
import pickle
import logging
from importlib import reload

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

        # where to save fiure and data
        self.base = {
            "title": "",
            "sub_folder": "",
            "data_folder": "data/",
            "ignore_while_saving": [],
        }

        logger.info("(%s) ... BaseClass initialized.", self._name)

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
            title = f"{self.base["title"]}.pickle"

        # Handle data folder
        folder = os.path.join(os.getcwd(), self.base["data_folder"], self.base["sub_folder"])
        check = os.path.isdir(folder)
        if not check and self.base["data_folder"] != "":
            os.makedirs(folder)

        # Get Dictionary
        data = {}
        for key, value in self.__dict__.items():
            if key not in self.base["ignore_while_saving"]:
                data[key] = value

        # save data to pickle
        name = os.path.join(os.getcwd(), self.base["data_folder"], self.base["sub_folder"], title)
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
            title = f"{self.base["title"]}.pickle"

        # get data from pickle
        name = os.path.join(os.getcwd(), self.base["data_folder"], self.base["sub_folder"], title)
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
