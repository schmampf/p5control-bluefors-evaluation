from enum import Enum

import logging

from integration.files import DataCollection

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

glob_indent_lvl: int = 0
local_indent_lvl: int = 0
anchor: list[int] = []
bib: DataCollection = DataCollection()
suppress_print: bool = False


class LogLevel(Enum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    NOTSET = logging.NOTSET


CRITICAL = LogLevel.CRITICAL
ERROR = LogLevel.ERROR
WARNING = LogLevel.WARNING
INFO = LogLevel.INFO
DEBUG = LogLevel.DEBUG


def setup(nbib: DataCollection) -> None:
    global bib
    bib = nbib
    print(INFO, START, "Logger.setup()")


def set_level(level: LogLevel) -> None:
    """Set the logging level."""
    logger.setLevel(level.value)
    print(INFO, START, "Logger.set_level()")


class Indent(Enum):
    START = 0
    KEEP = 1
    END = 2
    INC = 3
    RESET = 4


START = Indent.START
KEEP = Indent.KEEP
END = Indent.END
INC = Indent.INC
RESET = Indent.RESET


def print(plevel: LogLevel, nextIndent: Indent = KEEP, msg: str = "") -> None:
    """Print a message with the specified logging level and indentation."""
    global glob_indent_lvl
    global local_indent_lvl
    global anchor
    global data
    global suppress_print

    if suppress_print:
        return

    # if cascade:
    #     temp = last_anchor.pop() if last_anchor else 0
    #     last_anchor.append(glob_indent_lvl)
    #     glob_indent_lvl = temp

    if nextIndent == Indent.START:
        glob_indent_lvl = 0

    # if nextIndent == Indent.INC:
    #     glob_indent_lvl += 2

    # if nextIndent == Indent.RESET:
    #     glob_indent_lvl = last_anchor.pop() if last_anchor else 0

    name = bib.data.name if not bib.data.name == "default" else "..."
    logger.log(plevel.value, "(%s) %s%s", name, " " * glob_indent_lvl, msg)

    # if plevel == LogLevel.DEBUG:
    #     glob_indent_lvl += 2
    # else:
    glob_indent_lvl = 2

    if nextIndent == Indent.END:
        glob_indent_lvl = 0
