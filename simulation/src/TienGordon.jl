module TienGordon
import SpecialFunctions as SF
using JLD2
using Interpolations
export I

#-constants-------------
const e = 1.602176634e-19 # C
const ħ = 1.054571817e-34 # Js
#-----------------------
#-variables-------------
ν = 1.0
extend = false
I0 = []
V0 = []
n = 200
#-----------------------

const data_dir = joinpath(@__DIR__, "..", "data")

# find a value in y for a value in x
# if no direct match is found extrapolate from the closest
function findVal(query::Real, x::Array{Float64}, y::Array{Float64})
    for (i, val) in enumerate(x)
        if val < query
            continue
        elseif val == query
            return y[i]
        elseif val > query
            x_c, y_c = x[i], y[i]

            if i <= 1
                return y_c
            end

            x_p, y_p = x[i-1], y[i-1]
            x_diff = x_c - query
            y_diff = y_c - y_p
            return y_p + (y_diff * (x_diff / (x_c - x_p)))
        end
    end

    if query > x[end]
        return y[end]
    end

    throw(ArgumentError("Query value $query not found in x array."))
end

function set_τ(τ::Real)
    global V0, I0
    data = jldopen(data_dir * "/iv.jld2", "r")
    v, i = data["normalized"][τ]
    V0, I0 = v, i
    if extend
        V0 = vcat(reverse(V0) .* -1, V0)
        I0 = vcat(reverse(I0) .* -1, I0)
    end
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
    global ν, I0, V0, n

    ω = 2 * π * ν
    b_arg = (e * Vω) / (ħ * ω) / 10000

    out = 0.0
    for ni in -n:n
        bessel = (SF.besselj(ni, b_arg))^2
        V_shift = V₀ - (ni * ħ * ω) / (e)
        V_shift = abs(V_shift) # symmetry around the y axis
        I = findVal(V_shift, V0, I0)
        out += I * bessel
    end
    return out
end
end

