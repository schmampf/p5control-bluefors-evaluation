# This installs a new kernel, that is selectable from the Jupyter notebook interface.
using IJulia
installkernel("Julia (10 threads)", env=Dict("JULIA_NUM_THREADS" => "10"))
