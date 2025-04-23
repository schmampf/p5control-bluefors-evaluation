# region imports
# std lib
from enum import Enum
from typing import Any

# 3rd party

# local
import utilities.logging as Logger

# endregion


def name_generator(**kwargs) -> str:

    class Fields(Enum):
        title = "Untitled"
        variable = ("var", ("min", "max"), "unit")
        gate_voltage = "V"
        vna_frequency = "Hz"
        vna_amplitude = "V"
        motor_position = ""
        magnet = "T"

    title = kwargs.pop("title", Fields.title.value)
    variable = kwargs.pop("variable", "(None) (0,0)")
    constants = []
    for key_str, value in kwargs.items():
        try:
            name = Fields[key_str]
            # convert value to scientific notation if it's a float
            if isinstance(value, float):
                value = f"{value}"
            constants.append((key_str, value, name.value))
        except KeyError:
            Logger.print(
                Logger.ERROR, msg=f"KeyError: {key_str} is not a valid NameGen key."
            )
    var_str = f"{variable}".replace(" ", "").replace("'", "")
    const_str = f"{constants}".replace(" ", "").replace("'", "")
    return f"{title}: var={var_str} const={const_str}"
