# Definition for the custom snapshot error

struct SnapshotTimeError{N, T<:Real} <:Exception
    times::Tuple{T, Union{Nothing, T}}
end

SnapshotTimeError(time::Real) = SnapshotTimeError{1, typeof(time)}((time, nothing))
SnapshotTimeError(times::Vararg{Real, 2}) = SnapshotTimeError{2, eltype(times)}(times)

Base.showerror(io::IO, e::SnapshotTimeError{1}) = print(io, "Snapshot does not exist at: ", e.times[1])
Base.showerror(io::IO, e::SnapshotTimeError{2}) = print(io, "Snapshots do not exist between ", e.times[1], " - ", e.times[2])
