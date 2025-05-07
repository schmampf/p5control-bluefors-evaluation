import utilities.logging as Logger
import integration.files as Files

bib = Files.DataCollection()

Logger.setup(bib)
Logger.set_level(Logger.INFO)

bib.data.name = "test"

Logger.print(Logger.WARNING, msg="This is an info message")
