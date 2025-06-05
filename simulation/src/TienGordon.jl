module TienGordon
import SpecialFunctions as SF
using JLD2, Interpolations, Match
export I

#-constants-------------
const e = 1.602176634e-19 # C
const h = 6.62607015e-34 # Js
const ħ = h / (2 * π) # Js
#-----------------------
#-variables-------------
ν = 1.0
Δ = 1.0
τ = 0.0
n = 0
m = 1.0
reference = nothing
#-----------------------

const data_dir = joinpath(@__DIR__, "..", "data")

@enum ReferenceType single multiple
struct ReferenceDesc
    path::String
    set::String
end
struct Reference
    V0::Vector{<:Real}
    I0::Vector{<:Real}
    Im::Vector{Tuple{Vector{<:Real},Interpolations.Extrapolation}}
end
function load_reference(ref::ReferenceDesc; type::ReferenceType=single)
    global τ, reference
    @match type begin
        $single => begin
            data = jldopen(ref.path, "r")
            v, i = data[ref.set][τ]
            V0, I0 = v, i
            V0 = vcat(-reverse(V0), V0)
            I0 = vcat(-reverse(I0), I0)
            itp = linear_interpolation(V0, I0, extrapolation_bc=Line())
            reference = Reference(V0, I0, [(I0, itp)])
        end
        $multiple => begin
            data = jldopen(ref.path, "r")
            set = data[string(τ)]
            V0 = set["V0"]
            I0 = set["I0"]
            Im = set["Im"]

            Im = [Im[:, mi] for mi in axes(Im, 2)]

            if V0[1] != -V0[end]
                V0 = vcat(-reverse(V0), V0)
                I0 = vcat(-reverse(I0), I0)
                Im = [vcat(-reverse(Im[mi]), Im[mi]) for mi in 1:lastindex(Im)]
            end

            continuous(x) = begin
                stepsize = x[2] - x[1]
                state = true
                idx = 1
                while idx < lastindex(V0)
                    if V0[idx] + stepsize == V0[idx+1]
                        idx += 1
                    else
                        state = false
                        break
                    end
                end
                return state
            end
            fix(a, a_fix, b) = begin
                lookup = Dict(zip(Real.(round.(a, digits=2)), b))
                return [get(lookup, Real(round(x, digits=2)), 0) for x in a_fix]
            end

            if !continuous(V0)
                V0_fix = collect(V0[1]:V0[2]-V0[1]:V0[end])
                I0_fix = fix(V0, V0_fix, I0)
                Im_fix = [fix(V0, V0_fix, Im[mi]) for mi in 1:lastindex(Im)]
            end

            reference = Reference(V0, I0, [
                (Im_fix[mi], linear_interpolation(V0_fix, Im_fix[mi], extrapolation_bc=Line()))
                for mi in 1:lastindex(Im)
            ])
        end
    end
end

function save_curve(τ::Real)
    data = jldopen(data_dir * "/iv.jld2", "r")
    v, i = data["normalized"][τ]

    open(data_dir * "/iv_$τ.dat", "w") do file
        for j in 1:lastindex(v)
            println(file, v[j], "\t", i[j])
        end
    end
end

function IV₀(V₀::Real, Vω::Real)
    global ν, n, m, Δ, reference

    ω = 2 * π * ν
    b_arg = (e * Vω) / (ħ * ω)

    out = 0.0

    for mi in 1:m, ni in -n:n
        bessel = (SF.besselj(ni, mi * Vω))^2
        V_shift = V₀ - (ni * ħ * ω) / (mi * Δ * e)
        I = reference.Im[mi][2](V_shift)
        out += I * bessel
    end
    return out
end
end

