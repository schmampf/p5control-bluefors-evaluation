"""
The following simulation functions and formulars are taken and adapted from:
Cuevas et al., PRB (1996).
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.54.7366
"""
module Simulation
export maxMAR, cstm∫, VarPar

# export maxMAR
const π::Real = 4.0 * atan(1.0)

Base.@kwdef mutable struct VarPar
    max_ar::Int                     # The maximum number of MAR allowed by this simulation
    error::Real
    t_coeff::Real                   # The transmission coefficient
    temp::Real                      # The system temperature
    voltage::AbstractRange{<:Real}  # The voltages in units of the gap
    w::Tuple{Real,Real}             # integration limits

    function VarPar(
        max_ar::Int,
        error::Real,
        t_coeff::Real,
        temp::Real,
        voltage::AbstractRange{<:Real},
        w::Tuple{Real,Real}
    )
        temp = ifelse(temp <= 0.0, 1e-7, temp)
        new(max_ar, error, t_coeff, temp, voltage, w)
    end
end

function maxMAR(in::VarPar)
    num::Int = Int(2.0 / abs(in.voltage.start))
    num += ifelse(isodd(num), 6, 7)

    @assert num <= in.max_ar """ 
    The required number of MAR ($(num)) is greater than the currently allowed maximum ($(in.max_ar)).
    To fix this, either decrease the minimal voltage or increase the maximum number of MAR allowed.
    """
    return num
end

function cstm∫()
    # This function is a placeholder for the actual implementation
    # It should return a value based on the simulation parameters
    return 0.0
end

end

import .Simulation as Sim

params = Sim.VarPar(
    max_ar=520,
    error=1e-7,
    t_coeff=1e-7,
    temp=1e-7,
    voltage=range(start=0.05, stop=3.51, step=0.01),
    w=(-15.0, 15.0)
)

for i in 1:length(params.voltage)
    var = Int(round(2.0 / abs(params.voltage[i])))
    var += ifelse(isodd(var), 6, 7)
    current = Real(Sim.cstm∫())
end
