from enum import Enum

# import logging
from builtins import print as _print

from integration.files import DataCollection


# region logging levels
class Level(Enum):
    CRITICAL = 0
    ERROR = 1
    WARNING = 2
    INFO = 3
    DEBUG = 4
    NOTSET = 5


CRITICAL = Level.CRITICAL
ERROR = Level.ERROR
WARNING = Level.WARNING
INFO = Level.INFO
DEBUG = Level.DEBUG
# endregion


# region indentation
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
# endregion

bib: DataCollection = DataCollection()
suppress_print: bool = False
level: Level = Level.NOTSET

cache: str = ""

glob_indent_lvl: int = 0
# local_indent_lvl: int = 0
# anchor: list[int] = []


def setup(nbib: DataCollection):
    """Set up the logger with the given DataCollection."""
    global bib
    bib = nbib
    print(INFO, START, "Logger.setup()")


def set_level(nlevel: Level):
    """Set the logging level."""
    global level
    level = nlevel
    print(INFO, START, f"Logger.set_level(lvl={nlevel})")


def print(
    plevel: Level,
    nextIndent: Indent = KEEP,
    msg: str = "",
    force: bool = False,
    updating: bool = False,
):
    """Print a message with the specified logging level and indentation."""
    global level
    global glob_indent_lvl
    global suppress_print
    global cache

    if suppress_print and not force or plevel.value > level.value:
        return

    if nextIndent == START:
        glob_indent_lvl = 0

    indent = " " * glob_indent_lvl
    name = bib.data.name if not bib.data.name == "default" else "..."
    fmsg = f"({name}){indent} {msg}"

    if updating:
        _print(f"\r{len(cache) * ' '}", end="")
        _print(f"\r{fmsg}", end="")
    else:
        _print(fmsg)

    cache = fmsg

    glob_indent_lvl = 2

    if nextIndent == Indent.END:
        glob_indent_lvl = 0
