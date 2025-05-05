import utilities.logging as Logger
import integration.files as Files
import evaluation.general as GenEval
import evaluation.iv as IVEval
import plotting.plot as Plot
import utilities.macros as Macros

bib = Files.DataCollection()

Logger.setup(bib)
Logger.set_level(Logger.INFO)

Files.setup(bib, "test", "/home/dacap/Downloads")
bib.data.file_name = "OI-25c-09 2025-04-15 unbroken antenna full 1-formated.hdf5"

GenEval.setup(bib)
IVEval.setup(bib)
bib.params.volt_amp = (1.0, 1.0)

# Files.showFile(bib.data)

GenEval.loadMeasurements(bib)
GenEval.showLoadedMeasurements(bib)
GenEval.select_measurement(bib, 2)

IVEval.select_edge(bib, 1, "down")

Macros.macro_eval(bib, 1, "adwin")

# Plot.plot_curves(bib, ("cache", "adwin"), [("voltage", "current")], True)


# Plot.plot_map(bib, ("adwin/current", "adwin/voltage"))
