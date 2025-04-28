# Some Code Examples / Snippets

# =================================================
# Renaming groups in a h5py file
# -------------------------------------------------
# file = GenEval.File(Files.getfile_path(bib.data), "r+")
# oldPath = "measurement/vna_frequencies_0.200V"
# newpath = (
#     "measurement/Untitled: var=(vna_frequency,(0,0),Hz) const=[(vna_amplitude,0.2,V)]"
# )
# file.move(oldPath, newpath)
# ==================================================
# Generating a name with the NameGen module
# -------------------------------------------------
# label = name_generator(
#     # title="Fast_IV",
#     variable=("gate_voltage", (0.0, 1.0), "V"),
#     vna_frequency=4.31e9,
#     vna_amplitude=-0.5,
#     motor_position=0.0,
#     magnet=-0.0098989881,
# )
# ===================================================
# Recursivly copy the contents of a h5py file
# -------------------------------------------------
# from h5py import File, Group

# path = "/home/dacap/Downloads/OI-25c-09 2025-04-15 unbroken 0.hdf5"
# new_path = "/home/dacap/Downloads/OI-25c-09 2025-04-15 unbroken 0.copy.hdf5"

# # example data path: /measurement/Group/Subgroup/subsubgroup/dataset

# file = File(path, "r")
# dir = file.get("measurement")


# # copy stuff
# def copy_group(source: Group, dest: Group):
#     """
#     Recursively copy a group and its contents from source to destination.
#     """
#     for key, item in source.items():
#         if isinstance(item, Group):
#             new_group = dest.create_group(key)
#             copy_group(item, new_group)
#         else:
#             try:
#                 dest.copy(item, key)
#             except Exception as e:
#                 print(f"Error copying {key}: {e}")


# # Create a new file to copy the data into
# with File(new_path, "w") as new_file:
#     # Create a new group in the new file
#     new_group = new_file.create_group("measurement")

#     # Copy the contents of the original group to the new group
#     copy_group(dir, new_group)
# ===================================================
