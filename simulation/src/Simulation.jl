"""
The following simulation functions and formulars are taken and adapted from:
Cuevas et al., PRB (1996).
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.54.7366
"""
module Simulation
export num_MAR, ∫, VarPar
using Integrals
using OffsetArrays
using LinearAlgebra

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
    f(u, p) = integrand(in, u, curr_v, req_MAR, acc)
    domain = (in.w[1], in.w[2])
    prob = IntegralProblem(f, domain)
    sol = solve(prob, HCubatureJL(); reltol=1e-6, abstol=1e-8)

    return sol.u
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
    w::Real,
    curr_v::Real,
    req_MAR::Int,
    approx::Int=520
)
    T_fact::Real = in.t_coeff # The transmission coefficient
    τ::Matrix{Real} = [
        1.0 0.0;
        0.0 -1.0
    ]
    t_hopp::Real = sqrt((2.0 - T_fact - 2.0 * sqrt(1.0 - T_fact)) / T_fact)
    I::Matrix{Real} = [
        1.0 0.0;
        0.0 1.0
    ]

    G_ret = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)
    G_adv = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)
    G_comb = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)


    for j in -req_MAR-2:req_MAR+2
        ω_j = w + j * curr_v
        # due to symmetry the green's functions of the left and right junction part are the same
        G_ret[:, :, j] = green(ω_j, ret)
        G_adv[:, :, j] = green(ω_j, adv)
        fact = tanh(0.5 * ω_j / in.temp)
        G_comb[:, :, j] = (G_ret[:, :, j] - G_adv[:, :, j]) .* fact
    end

    # waste some memory as indexing in steps of 2 would be a pain
    e_ret = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)
    e_adv = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)
    v⁺_ret = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)
    v⁺_adv = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)
    v⁻_ret = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)
    v⁻_adv = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)

    shifted(M, (a, b, c))::Array{Complex} = [
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
        e_ret[:, :, j] = I - t_hopp^2 .* G_ret[:, :, j] .* shifted(G_ret, (1, 2, j))
        e_adv[:, :, j] = I - t_hopp^2 .* G_adv[:, :, j] .* shifted(G_adv, (1, 2, j))
        # for Eq. A3
        v⁺_ret[:, :, j] = mask_up .* t_hopp^2 .* G_ret[:, :, j+1] .* G_ret[:, :, j+2]
        v⁺_adv[:, :, j] = mask_up .* t_hopp^2 .* G_adv[:, :, j+1] .* G_adv[:, :, j+2]
        # for Eq. A4
        v⁻_ret[:, :, j] = mask_down .* t_hopp^2 .* G_ret[:, :, j-1] .* G_ret[:, :, j-2]
        v⁻_adv[:, :, j] = mask_down .* t_hopp^2 .* G_adv[:, :, j-1] .* G_adv[:, :, j-2]
    end

    aux_ret = Array{Complex}(undef, 2, 2, 4)
    aux_adv = Array{Complex}(undef, 2, 2, 4)
    aux_ret[:, :, 1] = copy(e_ret[:, :, req_MAR])
    aux_ret[:, :, 3] = copy(e_ret[:, :, -req_MAR])
    aux_ret[:, :, 2] = copy(inv(aux_ret[:, :, 1]))
    aux_ret[:, :, 4] = copy(inv(aux_ret[:, :, 3]))
    aux_adv[:, :, 1] = copy(e_adv[:, :, req_MAR])
    aux_adv[:, :, 3] = copy(e_adv[:, :, -req_MAR])
    aux_adv[:, :, 2] = copy(inv(aux_adv[:, :, 1]))
    aux_adv[:, :, 4] = copy(inv(aux_adv[:, :, 3]))

    ad_ret = OffsetArray(Array{Complex}(undef, 2, 2, req_MAR), 1:2, 1:2, 1:req_MAR)
    ad_adv = OffsetArray(Array{Complex}(undef, 2, 2, req_MAR), 1:2, 1:2, 1:req_MAR)
    ai_ret = OffsetArray(Array{Complex}(undef, 2, 2, req_MAR), 1:2, 1:2, -req_MAR:-1)
    ai_adv = OffsetArray(Array{Complex}(undef, 2, 2, req_MAR), 1:2, 1:2, -req_MAR:-1)

    ad_ret[:, :, req_MAR] = copy(aux_ret[:, :, 2])
    ad_adv[:, :, req_MAR] = copy(aux_adv[:, :, 2])
    ai_ret[:, :, -req_MAR] = copy(aux_ret[:, :, 4])
    ai_adv[:, :, -req_MAR] = copy(aux_adv[:, :, 4])

    for j = req_MAR-2:-2:1
        for k = 1:2, l = 1:2
            aux_ret[k, l, 1] = e_ret[k, l, j]
            aux_ret[k, l, 3] = e_ret[k, l, -j]
            aux_adv[k, l, 1] = e_adv[k, l, j]
            aux_adv[k, l, 3] = e_adv[k, l, -j]

            for n in 1:2, m in 1:2
                aux_ret[k, l, 1] -= v⁺_ret[k, n, j] * ad_ret[n, m, j+2] * v⁻_ret[m, l, j+2]
                aux_ret[k, l, 3] -= v⁻_ret[k, n, -j] .* ai_ret[n, m, -j-2] .* v⁺_ret[m, l, -j-2]
                aux_adv[k, l, 1] -= v⁺_adv[k, n, j] .* ad_adv[n, m, j+2] .* v⁻_adv[m, l, j+2]
                aux_adv[k, l, 3] -= v⁻_adv[k, n, -j] .* ai_adv[n, m, -j-2] .* v⁺_adv[m, l, -j-2]
            end
        end

        aux_ret[:, :, 2] = inv(aux_ret[:, :, 1])
        aux_ret[:, :, 4] = inv(aux_ret[:, :, 3])
        aux_adv[:, :, 2] = inv(aux_adv[:, :, 1])
        aux_adv[:, :, 4] = inv(aux_adv[:, :, 3])

        ad_ret[:, :, j] = copy(aux_ret[:, :, 2])
        ai_ret[:, :, -j] = copy(aux_ret[:, :, 4])
        ad_adv[:, :, j] = copy(aux_adv[:, :, 2])
        ai_adv[:, :, -j] = copy(aux_adv[:, :, 4])
    end

    t_x = Array{Complex}(undef, 2, 2)
    t_y = Array{Complex}(undef, 2, 2)
    t_x[:, :] .= 0.0
    t_y[:, :] .= 0.0
    t_x[2, 2] = t_hopp
    t_y[1, 1] = t_hopp

    c⁺_ret = Array{Complex}(undef, 2, 2)
    c⁺_adv = Array{Complex}(undef, 2, 2)
    c⁻_ret = Array{Complex}(undef, 2, 2)
    c⁻_adv = Array{Complex}(undef, 2, 2)

    for k = 1:2, l = 1:2
        aux_ret[k, l, 1] = e_ret[k, l, 1]
        aux_adv[k, l, 1] = e_adv[k, l, 1]
        c⁺_ret[k, l] = t_x[k, l]
        c⁺_adv[k, l] = t_x[k, l]

        for m = 1:2, n = 1:2
            aux_ret[k, l, 1] -= v⁺_ret[k, m, 1] * ad_ret[m, n, 3] * v⁻_ret[n, l, 3] - v⁻_ret[k, m, 1] * ai_ret[m, n, -1] * v⁺_ret[n, l, -1]
            aux_adv[k, l, 1] -= v⁺_adv[k, m, 1] * ad_adv[m, n, 3] * v⁻_adv[n, l, 3] - v⁻_adv[k, m, 1] * ai_adv[m, n, -1] * v⁺_adv[n, l, -1]

            c⁺_ret[k, l] += v⁻_ret[k, m, 1] * ai_ret[m, n, -1] * t_y[n, l]
            c⁺_adv[k, l] += v⁻_adv[k, m, 1] * ai_adv[m, n, -1] * t_y[n, l]
        end
    end

    aux_ret[:, :, 2] = inv(aux_ret[:, :, 1])
    aux_adv[:, :, 2] = inv(aux_adv[:, :, 1])

    t_ret = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)
    t_adv = OffsetArray(Array{Complex}(undef, 2, 2, 2 * approx + 1), 1:2, 1:2, -approx:approx)

    for k = 1:2, l = 1:2
        t_ret[k, l, 1] = 0.0
        t_adv[k, l, 1] = 0.0
        for m = 1:2
            t_ret[k, l, 1] += aux_ret[k, m, 2] * c⁺_ret[m, l]
            t_adv[k, l, 1] += aux_adv[k, m, 2] * c⁺_adv[m, l]
        end
    end

    for k = 1:2, l = 1:2
        c⁻_ret[k, l] = t_y[k, l]
        c⁻_adv[k, l] = t_y[k, l]
        for m = 1:2
            c⁻_ret[k, l] += v⁺_ret[k, m, -1] * t_ret[m, l, 1]
            c⁻_adv[k, l] += v⁺_adv[k, m, -1] * t_adv[m, l, 1]
        end
    end

    for k = 1:2, l = 1:2
        t_ret[k, l, -1] = 0.0
        t_adv[k, l, -1] = 0.0
        for m = 1:2
            t_ret[k, l, -1] += ai_ret[k, m, -1] * c⁻_ret[m, l]
            t_adv[k, l, -1] += ai_adv[k, m, -1] * c⁻_adv[m, l]
        end
    end

    for j = 3:2:req_MAR
        for k = 1:2, l = 1:2
            t_ret[k, l, j] = 0.0
            t_ret[k, l, -j] = 0.0
            t_adv[k, l, j] = 0.0
            t_adv[k, l, -j] = 0.0
            for m = 1:2, n = 1:2
                t_ret[k, l, j] += ad_ret[k, m, j] * v⁻_ret[n, m, j] * t_ret[n, l, j-2]
                t_ret[k, l, -j] += ai_ret[k, m, -j] * v⁺_ret[n, m, -j] * t_ret[n, l, -j+2]
                t_adv[k, l, j] += ad_adv[k, m, j] * v⁻_adv[n, m, j] * t_adv[n, l, j-2]
                t_adv[k, l, -j] += ai_adv[k, m, -j] * v⁺_adv[n, m, -j] * t_adv[n, l, -j+2]
            end
        end
    end

    curr::Real = 0.0
    for j = -req_MAR:2:req_MAR
        for k = 1:2, l = 1:2, m = 1:2, n = 1:2
            curr += real(
                τ[k, k] * t_ret[k, l, j] * G_comb[l, m, 0] *
                τ[m, m] * conj(t_ret[n, m, j]) *
                τ[n, n] * G_adv[n, k, j]
                +
                τ[k, k] * G_ret[k, l, 0] *
                τ[l, l] * conj(t_adv[m, l, j]) *
                τ[m, m] * G_comb[m, n, j] * t_adv[n, k, j]
            )
        end
    end
    curr /= 2.0

    return curr
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

