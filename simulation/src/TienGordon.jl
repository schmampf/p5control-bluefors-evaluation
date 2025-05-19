module TG
using Bessels
export I
function I(V₀::Real, Vω::Real, acc::Int=500)
    γ = 0.0051
    ν = 7.8E9 # GHz

    Vω = γ * Vω
    e = 1.602e-19 # C
    ħ = 1.055e-34 # Js
    ω = 2 * π * ν # rad/s
    I₀ = 1.0

    range = collect(-acc:1:acc)
    f(n) = (besselj(n, (e * Vω) / (ħ * ω)))^2 * I₀ * (V₀ - (n * ħ * ω) / (e))
    return sum(f.(range))
end
end
using .TG
I(-5, 0.5)