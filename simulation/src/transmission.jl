"""
The following simulation functions and formulars are taken and adapted from:
Cuevas et al., PRB (1996).
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.54.7366
"""
module Simulation
export num_MAR, ∫, VarPar
using Integrals

const π::Real = 4.0 * atan(1.0)
const η::Real = 0.000001
const Δ::Real = 1.0

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

function num_MAR(voltage::Real, param::VarPar)
    num::Int = Int(round(2.0 / abs(voltage)))
    num += ifelse(isodd(num), 6, 7)

    @assert num <= param.max_ar """ 
    The required number of MAR ($(num)) is greater than the currently allowed maximum ($(param.max_ar)).
    To fix this, either decrease the minimal voltage or increase the maximum number of MAR allowed.
    """
    return num
end

function ∫(in::VarPar, curr_v::Real, req_MAR::Int, acc::Int=520)
    # using Integrals
    # f(u, p) = sum(sin.(u))
    # domain = (ones(3), 3ones(3)) # (lb, ub)
    # prob = IntegralProblem(f, domain)
    # sol = solve(prob, HCubatureJL(); reltol = 1e-3, abstol = 1e-3)
    # sol.u
    return 0.0
end

"""
Compute a single integrand of the current integral. (See Eq. ? in Cuevas et al.)
# Arguments
- `in::VarPar`: A collection of parameters for the simulation.
- `curr_v::Real`: The current voltage that the integral is evaluated for.
- `req_MAR::Int`: A dynamically calculated number of MAR that is required for the current voltage.
- `approx::Int`: The number of points to use for the approximation.
"""
function integrand(
    in::VarPar,
    curr_v::Real,
    req_MAR::Int,
    approx::Int=520
)
    llim::Real = in.w[1] # upper integration limit
    ulim::Real = in.w[2] # lower integration limit
    T_fact::Real = in.t_coeff # The transmission coefficient
    τ::Matrix{Real} = [
        [1.0, 0.0],
        [0.0, -1.0]
    ]
    t_hopp::Real = sqrt((2.0 - in.t_coeff - 2.0 * sqrt(1.0 - in.t_coeff)) / in.t_coeff)

    G_ret::Matrix{Complex} = (undef, 2, 2, 2 * approx + 5)
    G_adv::Matrix{Complex} = (undef, 2, 2, 2 * approx + 5)
    G_comb::Matrix{Complex} = (undef, 2, 2, 2 * approx + 5)

    for AR_j in -req_MAR-2:req_MAR+2
        ω_j = AR_j * curr_v
        # due to symmetry the green's functions of the left and right junction part are the same
        G_ret[:, :, AR_j] = green(ω_j, ret)
        G_adv[:, :, AR_j] = green(ω_j, adv)
        fact = tanh(0.5 * ω_j / in.temp)
        G_comb[:, :, AR_j] = (G_ret[:, :, AR_j] - G_adv[:, :, AR_j]) * fact
    end


end

@enum Sign ret adv

"""
Calculated the uncoupled -retarded and -advanced Green's function.
(See Paper Eq. 8)
# Arguments
- `ω_j::Real`: The frequency at which the Green's function is evaluated.
- `type::Sign`: The type of Green's function to calculate
    - `ret`: for a retarded Green's function (addition of components)
    - `adv`: for an advanced Green's function (subtraction of components)
"""
function green(ω_j::Real, typ::Sign)
    op = ifelse(typ == ret, +, -)

    W = 1.0
    ω = op(ω_j, (im * η))
    fact = 1 / (W * sqrt(Δ^2 - ω^2))

    return fact * [-ω Δ; Δ -ω]
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
currents::Vector{Real} = zeros(length(params.voltage))

# check max_MAR
Sim.num_MAR(first(params.voltage), params)

for i in 1:length(params.voltage)
    curr_v::Real = params.voltage[i]
    req_MAR::Int = Sim.num_MAR(curr_v, params)
    currents[i] = Real(Sim.∫(params, curr_v, req_MAR))
end
