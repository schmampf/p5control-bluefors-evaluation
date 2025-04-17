import utilities.logging as Logger
import integration.files as Files
import evaluation.general as GenEval

# bib = Files.DataCollection()

# Logger.setup(bib)
# Logger.set_level(Logger.INFO)

# Files.setup(bib, "test", "/home/dacap/Downloads")
# bib.data.file_name = "23_11_15_PR22e9_4.3G_1.hdf5"

# GenEval.setup(bib)
# Files.showFile(bib.data)
# GenEval.loadMeasurements(bib)
# GenEval.showLoadedMeasurements(bib)
# GenEval.select_measurement(bib, 10)


from enum import Enum
from typing import Any


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


label = name_generator(
    # title="Fast_IV",
    variable=("gate_voltage", (0.0, 1.0), "V"),
    vna_frequency=4.31e9,
    vna_amplitude=-0.5,
    motor_position=0.0,
    magnet=-0.0098989881,
)

# header = GenEval.MeasurementHeader.from_string(label)

print(label)
# print(header.to_string())
