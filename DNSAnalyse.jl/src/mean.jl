# Utility functions. to compute the mean profile

function _mean!(ū::Vector{Float64}, data::DNSData{<:Any, Nz}, snap_times::Vector{Float64}) where {Nz}
    # loop over the time window of the data and compute mean
    for t in snap_times
        ū .+= dropdims(sum(data[t][1], dims=2), dims=2)./Nz
    end
    ū ./= length(snap_times)

    # add back the laminar profile
    ū .+= data.y

    return ū
end

function mean!(ū, data::DNSData; window::NTuple{2, Real}=(firstindex(data), lastindex(data)))
    # find range of snapshots inside the provided window
    snapshot_times = data.snaps
    start_ti = findfirst(x->window[1]<=x, snapshot_times)
    end_ti = findlast(x->window[2]>=x, snapshot_times)

    # overwrite the mean with zeros
    ū .= zero(Float64)

    # compute mean
    return _mean!(ū, data, snapshot_times[start_ti:end_ti])
end

mean(data::DNSData{Ny}; window::NTuple{2, Real}=(firstindex(data), lastindex(data))) where {Ny} = (ū = zeros(Float64, Ny); mean!(ū, data, window=window))
