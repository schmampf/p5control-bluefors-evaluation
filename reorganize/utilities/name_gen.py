# region imports
# std lib
from enum import Enum
from typing import Any

# 3rd party

# local
import utilities.logging as Logger

# endregion

# def format_number(value: float) -> str:
#     # format a number to its scientific notation
#     # 1GHz -> 1e9
#     # 1.0V -> 1e0
#     #13.7GHz -> 1.37e10

#     num_str = str(value)

prefix: dict[str, int] = {
    "n": -9,
    "u": -6,
    "m": -3,
    "k": 3,
    "M": 6,
    "G": 9,
}


def format_unit(unit: str):
    # V -> (1e0, "V")
    if len(unit) < 2:
        return (1e0, unit)

    # GHz -> (1e9, "Hz")
    if unit[0] in prefix.keys():
        prefix_value = prefix[unit[0]]
        unit = unit[1:]
        value = 10**prefix_value
        return (value, unit)
    else:
        prefix_value = 0
        unit = unit[1:]
        value = 10**prefix_value
        return (value, unit)


def format_scientific(s):
    s = format(s, ".15e")
    base, exp = s.split("e")
    base = base.rstrip("0").rstrip(".")  # strip trailing zeros
    return f"{base}e{exp}"


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
    if title == "Untitled":
        title = ""
    else:
        title += ": "

    var_name, var_range, var_unit = kwargs.pop("variable", Fields.variable.value)
    var_unit = format_unit(var_unit)
    var_range = [format_scientific(x * var_unit[0]) for x in var_range]

    variable = (var_name, tuple(var_range), var_unit[1])

    constants = []
    for key_str, value in kwargs.items():
        try:
            name = Fields[key_str]
            # convert value to scientific notation if it's a float
            if isinstance(value, float):
                value = f"{value}"
            constants.append((key_str, format_scientific(float(value)), name.value))
        except KeyError:
            Logger.print(
                Logger.ERROR, msg=f"KeyError: {key_str} is not a valid NameGen key."
            )
    var_str = f"{variable}".replace(" ", "").replace("'", "")
    const_str = f"{constants}".replace(" ", "").replace("'", "")
    return f"{title}var={var_str} const={const_str}"
