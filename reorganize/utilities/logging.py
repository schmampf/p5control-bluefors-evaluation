from enum import Enum

import logging

from integration.files import DataCollection

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

indent_level: int = 0
bib: DataCollection = DataCollection()

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

START = Indent.START
KEEP = Indent.KEEP
END = Indent.END

def print(level: LogLevel, nextIndent: Indent=KEEP, msg: str = "") -> None:
    """Print a message with the specified logging level and indentation."""
    global indent_level
    global data
    
    if nextIndent == Indent.START:
        indent_level = 0
    
    name = bib.data.name if not bib.data.name == "default" else "..."
    logger.log(level.value, "(%s) %s%s", name, " " * indent_level, msg)
    
    indent_level = 2
    
    if nextIndent == Indent.END:
        indent_level = 0
