# This installs a new kernel, that is selectable from the Jupyter notebook interface.
ENV["JUPYTER"] = "/usr/bin/jupyter"
using IJulia
installkernel("Julia (12 threads)", env=Dict("JULIA_NUM_THREADS" => "12"))
