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
