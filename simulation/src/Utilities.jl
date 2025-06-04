module Utilities
export local_average, deriv1, deriv2
using Statistics: median

deriv1 = Vector{<:Real}
deriv2 = Vector{<:Real}
deriv3 = Vector{<:Real}

function moving_avg(data::Vector{<:Real}, window_size::Int=6)
    out = copy(data)
    di = Int(round(window_size / 2))
    for i in data
        if i > window_size && i < length(data) - window_size
            seg = data[(i-di):(i+di)]
            out[i] = mean(seg)
        end
    end
    return out
end

function local_average(data::Tuple{Vector{<:Real},Vector{<:Real}}; threshold::Real=0.1, window_size::Int=6)
    global deriv1, deriv2, deriv3

    x = data[1]
    y = data[2]

    out = copy(y)

    d1 = abs.(diff(y) ./ diff(x))
    d2 = abs.(diff(d1) ./ diff(x[1:end-1]))
    d3 = abs.(diff(d2) ./ diff(x[1:end-2]))
    deriv1 = d1
    deriv2 = d2
    deriv3 = d3

    spots = findall(d2 .> threshold)
    filter!(i -> !(x[i] < 0.75) && !(0.95 < x[i] < 1.05) && !(1.9 < x[i] < 2.1), spots)
    #interpolate with window around the spikes
    di = Int(round(window_size / 2))
    for i in spots
        if i > window_size && i < length(y) - window_size
            seg = y[(i-di):(i+di)]
            dseg = diff(seg)
            med_seg = median(dseg)
            for j in 1:lastindex(seg)
                out[i+j] = out[i] + j * med_seg
            end
        else
            out[i] = y[i]
        end
    end

    # find index where x is >= 2.25
    idx_start = lastindex(x)
    for (i, val) in enumerate(x)
        if val >= 2.25
            idx_start = i
            break
        end
    end
    if idx_start > 0
        out[idx_start:end] .= moving_avg(out[idx_start:end], 11)
    end
    return out
end
end