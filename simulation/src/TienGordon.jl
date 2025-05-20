module TienGordon
using Bessels
using JLD2
export I

#-constants-------------
const e = 1.602e-19 # C
const ħ = 1.055e-34 # Js
#-----------------------
#-variables-------------
ν = 1.0
extend = false
#-----------------------

const data_dir = joinpath(@__DIR__, "..", "data")

# find a value in y for a value in x
# if no direct match is found extrapolate from the closest
function findVal(exact::Real, x::Array{Float64}, y::Array{Float64})
    for (i, val) in enumerate(x)
        if val == exact
            return y[i]
        elseif val < exact
            continue
        elseif val > exact
            x_c, y_c = x[i], y[i]

            if i < 2
                return y_c
            else
                return y[i-1]
            end

            # x_p, y_p = x[i-1], y[i-1]
            # x_diff = x_c - exact
            # y_diff = y_c - y_p
            # y_exact = y_p + (y_diff * (x_diff / (x_c - x_p)))
            # return y_exact
        end
    end
    # println("Warning: requested $exact")
    return y[end]
end

function V0()
    data = jldopen(data_dir * "/iv.jld2", "r")
    dict = data["raw"]
    key1 = first(keys(dict))
    V0 = dict[key1][1]
    if extend
        V0 = vcat(reverse(V0) .* -1, V0)
    end
    return V0
end

function IV₀(V₀::Real, Vω::Real, n::Int, τ::Real)
    global ν
    data = jldopen(data_dir * "/iv.jld2", "r")
    ref_V0, ref_I0 = data["normalized"][τ]
    ω = 2 * π * ν


    # special case
    if n == 0
        b_arg = (e * Vω) / (ħ * ω)
        bessel = (besselj(0, b_arg))^2
        V_shift = V₀
        I = findVal(V_shift, ref_V0, ref_I0)
        return I * bessel
    end

    # default case
    out = 0.0
    for ni in -n:n
        b_arg = (e * Vω) / (ħ * ω)
        bessel = (besselj(ni, b_arg))^2
        V_shift = V₀ - (ni * ħ * ω) / (e)
        I = findVal(V_shift, ref_V0, ref_I0)
        out += I * bessel
    end
    return out
end
end

