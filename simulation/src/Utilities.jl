module Utilities
export local_average

function local_average(data::Vector{<:Real}; threshold::Real=0.1, window_size::Int=6)
    out = copy(data)
    dx = diff(data)
    dx2 = abs.(diff(dx))
    spots = findall(dx2 .> threshold)
    #interpolate with window around the spikes
    di = Int(round(window_size / 2))
    for i in spots
        if i > window_size && i < length(data) - window_size
            seg = data[(i-di):(i+di)]
            dseg = diff(seg)
            med_seg = median(dseg)
            for j in 1:lastindex(seg)
                out[i+j] = out[i] + j * med_seg
            end
        else
            out[i] = data[i]
        end
    end

    return out
end
end