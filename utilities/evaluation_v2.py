"""Module providing a function printing python version."""

import logging
from importlib import reload

from utilities.baseplotting import BasePlotting

reload(logging)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


class EvaluationScript(BasePlotting):
    """class doc string."""

    def __init__(
        self,
        name="eva",
    ):
        self._name = name

        super().__init__()
