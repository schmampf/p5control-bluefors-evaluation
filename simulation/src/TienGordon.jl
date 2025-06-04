module TienGordon
import SpecialFunctions as SF
using JLD2
using Interpolations
export I

#-constants-------------
const e = 1.602176634e-19 # C
const h = 6.62607015e-34 # Js
const ħ = h / (2 * π) # Js
#-----------------------
#-variables-------------
ν = 1.0
Δ = 1.0
I0 = []
V0 = []
n = 0
m = 1.0
itp = nothing
reference = nothing
#-----------------------

const data_dir = joinpath(@__DIR__, "..", "data")

function set_τ(τ::Real)
    global V0, I0, itp
    data = jldopen(reference[1], "r")
    v, i = data[reference[2]][τ]
    V0, I0 = v, i

    V0 = vcat(-reverse(V0), V0)
    I0 = vcat(-reverse(I0), I0)

    itp = linear_interpolation(V0, I0, extrapolation_bc=Line())

    return τ
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
    global ν, I0, V0, n, m, Δ

    ω = 2 * π * ν
    b_arg = (e * Vω) / (ħ * ω)

    out = 0.0

    for ni in -n:n
        bessel = (SF.besselj(ni, Vω))^2
        V_shift = V₀ - (ni * ħ * ω) / (Δ * e)
        I = itp(V_shift)
        out += I * bessel
    end
    return out
end
end

