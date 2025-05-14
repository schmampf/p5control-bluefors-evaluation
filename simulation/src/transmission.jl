"""
The following simulation functions and formulars are taken and adapted from:
Cuevas et al., PRB (1996).
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.54.7366
"""
module Simulation
export num_MAR, ∫, VarPar
using Integrals
using OffsetArrays

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
    integrand(in, curr_v, req_MAR, acc)
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
        1.0 0.0;
        0.0 -1.0
    ]
    t_hopp::Real = sqrt((2.0 - in.t_coeff - 2.0 * sqrt(1.0 - in.t_coeff)) / in.t_coeff)
    I::Matrix{Real} = [
        1.0 0.0;
        0.0 1.0
    ]

    G_ret::Array{Complex} = Array{Complex}(undef, 2, 2, 2 * approx + 5)
    G_adv::Array{Complex} = Array{Complex}(undef, 2, 2, 2 * approx + 5)
    G_comb::Array{Complex} = Array{Complex}(undef, 2, 2, 2 * approx + 5)

    g_ret = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), (0, 0, -approx - 1))

    for j in -req_MAR-2:req_MAR+2
        ω_j = j * curr_v
        j = j + req_MAR + 2 + 1
        # due to symmetry the green's functions of the left and right junction part are the same
        G_ret[:, :, j] = green(ω_j, ret)
        G_adv[:, :, j] = green(ω_j, adv)
        fact = tanh(0.5 * ω_j / in.temp)
        G_comb[:, :, j] = (G_ret[:, :, j] - G_adv[:, :, j]) * fact
    end

    e_ret::Array{Complex} = Array{Complex}(undef, 2, 2, approx + 1)
    e_adv::Array{Complex} = Array{Complex}(undef, 2, 2, approx + 1)
    v⁺_ret::Array{Complex} = Array{Complex}(undef, 2, 2, approx + 1)
    v⁺_adv::Array{Complex} = Array{Complex}(undef, 2, 2, approx + 1)
    v⁻_ret::Array{Complex} = Array{Complex}(undef, 2, 2, approx + 1)
    v⁻_adv::Array{Complex} = Array{Complex}(undef, 2, 2, approx + 1)

    shifted(M::Array{Complex}, (a, b, c))::Array{Complex} = [
        M[a, b, c+1] M[a, b, c+1];
        M[a, b, c-1] M[a, b, c-1]
    ]

    mask_up::Matrix{Real} = [
        1.0 1.0;
        0.0 0.0
    ]
    mask_down::Matrix{Real} = [
        0.0 0.0;
        1.0 1.0
    ]

    for j in -req_MAR:2:req_MAR
        # for Eq. A2
        e_ret[:, :, j] = I - t_hopp^2 * G_ret[:, :, j] * shifted(G_ret[:, :, j], (1, 2, j))
        e_adv[:, :, j] = I - t_hopp^2 * G_adv[:, :, j] * shifted(G_adv[:, :, j], (1, 2, j))
        # for Eq. A3
        v⁺_ret[:, :, j] = mask_up * t_hopp^2 * G_ret[:, :, j+1] * G_ret[:, :, j+2]
        v⁺_adv[:, :, j] = mask_up * t_hopp^2 * G_adv[:, :, j+1] * G_adv[:, :, j+2]
        # for Eq. A4
        v⁻_ret[:, :, j] = mask_down * t_hopp^2 * G_ret[:, :, j-1] * G_ret[:, :, j-2]
        v⁻_adv[:, :, j] = mask_down * t_hopp^2 * G_adv[:, :, j-1] * G_adv[:, :, j-2]
    end

    aux_ret::Matrix{Complex} = Matrix{Complex}(undef, 2, 2, 4)
    aux_adv::Matrix{Complex} = Matrix{Complex}(undef, 2, 2, 4)
    aux_ret[:, :, 1] = e_ret[:, :, req_MAR]
    aux_ret[:, :, 2] = inv.(aux_ret[:, :, 1])
    aux_ret[:, :, 3] = e_ret[:, :, 1]
    aux_ret[:, :, 4] = inv.(aux_ret[:, :, 3])
    aux_adv[:, :, 1] = e_adv[:, :, req_MAR]
    aux_adv[:, :, 2] = inv.(aux_adv[:, :, 1])
    aux_adv[:, :, 3] = e_adv[:, :, 1]
    aux_adv[:, :, 4] = inv.(aux_adv[:, :, 3])

    # ad_* is indexed from 1 to req_MAR
    ad_ret::Matrix{Complex} = Matrix{Complex}(undef, 2, 2, req_MAR)
    ad_adv::Matrix{Complex} = Matrix{Complex}(undef, 2, 2, req_MAR)
    # ai_* is indexed from -req_MAR to -1
    ai_ret::Matrix{Complex} = Matrix{Complex}(undef, 2, 2, req_MAR)
    ai_adv::Matrix{Complex} = Matrix{Complex}(undef, 2, 2, req_MAR)

    ad_ret[:, :, req_MAR] = aux_ret[:, :, 2]
    ad_adv[:, :, req_MAR] = aux_adv[:, :, 2]
    ai_ret[:, :, req_MAR] = aux_ret[:, :, 4]
    ai_adv[:, :, req_MAR] = aux_adv[:, :, 4]

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
function green(ω_j::Real, typ::Sign)::Matrix{Complex}
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
