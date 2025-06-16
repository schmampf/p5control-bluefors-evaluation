import utilities.logging as Logger
import integration.files as Files
import evaluation.general as GenEval
import utilities.name_gen as NameGen

import h5py


# region functions
def rename_header(path: str, old: str, new: str):
    oldPath = path + old
    newPath = path + new
    try:
        file.move(oldPath, newPath)
    except Exception as e:
        print(f"Error renaming {oldPath} to {newPath}: {e}")
    # print(f"Renaming: \n\t {oldPath} \n\t {newPath}")


def sub_rename(path: str, old_pattern: str, new_pattern: str, back_cut: str = ""):
    group = file.get(path)
    if group and isinstance(group, h5py.Group):
        for name in group.keys():
            name = name.replace(back_cut, "")
            old_name = name
            new_name = name

            if "no_" in name or "NAN" in name:
                if back_cut in name:
                    old_path = path + "/" + name + back_cut
                else:
                    old_path = path + "/" + name
                new_name = new_pattern + "NAN"
            elif old_pattern in name:
                old_path = path + "/" + name + back_cut
                diff = name.replace(old_pattern, "")

                val, unit = GenEval.MeasurementHeader.parse_number(diff)
                unit = NameGen.format_unit(unit)
                val = NameGen.format_scientific(float(val) * unit[0])
                new_name = new_pattern + val + unit[1]

            new_path = path + "/" + new_name

            if old_path == new_path or old_name == new_name:
                # print(f"Skipping: \n\t {old_path} \n\t {new_path}")
                continue

            try:
                print(f"Renaming: \n\t {old_name} \n\t {new_name}")
                file.move(old_path, new_path)
            except Exception as e:
                print(f"Error renaming {old_name} to {new_name}: {e}")


def rename(
    path: str,
    old: str,
    new: str,
    old_pattern: str,
    new_pattern: str,
    back_cut: str = "",
):
    rename_header(path, old, new)
    if not old_pattern == new_pattern:
        sub_rename(path + new, old_pattern, new_pattern, back_cut)


# endregion

bib = Files.DataCollection()

Logger.setup(bib)
Logger.set_level(Logger.INFO)

Files.setup(bib, "test", "/home/dacap/Downloads")
bib.data.file_name = "2023-11-04_G0_antenna.hdf5"

file = GenEval.File(Files.getfile_path(bib.data), "r+")


# rename(
#     "/measurement/",
#     "vna_amplitudes_7.8000GHz",
#     NameGen.name_generator(
#         variable=("vna_amplitudes", (0.01, 1.0), "V"), vna_frequency=7.8e9
#     ),
#     "vna_7.8000GHz_",
#     "V=",
# )

rename(
    "/measurement/",
    "frequency_at_15GHz",
    NameGen.name_generator(
        variable=("vna_amplitudes", (-31.0, 0.0), "dBm"), vna_frequency=15e9
    ),
    "",
    "",
)

Files.showFile(bib.data)
