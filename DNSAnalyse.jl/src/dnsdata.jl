# Definitions for the top level interface for the DNS data set.

struct DNSData{Ny, Nz, Nt}
    loc::String
    params::Inifile
    snaps::Vector{Float64}
end

function DNSData(loc::String)
    ini = _read_params(loc*"params")
    snaps = sort(tryparse.(Float64, _filterDirectoryToSnapshots(loc)))
    DNSData{_fetch_param(ini, :Ny, Int), _fetch_param(ini, :Nz, Int), length(snaps)}(loc, ini, snaps)
end
loadDNS(loc) = DNSData(string(loc))

function Base.getproperty(data::DNSData{Ny, Nz, Nt}, field::Symbol) where {Ny, Nz, Nt}
    if field === :Ny
        return Ny
    elseif field === :Nz
        return Nz
    elseif field === :Nt
        return Nt
    elseif field ∈ [:Re, :Ro, :L, :dt, :T, :n_it_out, :t_restart, :stretch_factor, :n_threads]
        return _getparamfield(data, field)
    elseif field === :β
        return 2π/_getparamfield(data, :L)
    elseif field === :ω
        return 2π/_getparamfield(data, :T)
    elseif field === :dt_snap
        return _getparamfield(data, :n_it_out)*_getparamfield(data, :dt)
    elseif field === :y
        step = 1/(Ny - 1)
        sf = getproperty(data, :stretch_factor)
        return tanh.(sf*((0:Ny - 1)*step .- 0.5))/tanh(0.5*sf)
    else
        return getfield(data, field)
    end
end

Base.iterate(data::DNSData{Ny, Nz, Nt}, state::Int=1) where {Ny, Nz, Nt} = state > Nt ? nothing : (Snapshot(data.loc*@sprintf("%.6f", data.snaps[state])*"/", Ny, Nz), state + 1)
Base.eltype(::Type{DNSData}) = Snapshot
Base.eltype(::DNSData) = eltype(DNSData)
Base.length(::DNSData{<:Any, <:Any, Nt}) where {Nt} = Nt
Base.size(::DNSData{Ny, Nz, Nt}) where {Ny, Nz, Nt} = (Ny, Nz, Nt)

function Base.getindex(data::DNSData{Ny, Nz}, t::Real) where {Ny, Nz}
    i = findfirst(x->x==t, data.snaps)
    isnothing(i) ? throw(SnapshotTimeError(t)) : Snapshot(data.loc*@sprintf("%.6f", data.snaps[i])*"/", Ny, Nz)
end
Base.getindex(data::DNSData, ::Nothing) = data
Base.getindex(data::DNSData, range::NTuple{2, Real}; skip_step::Int=1) = getindex(data, range..., skip_step=skip_step)
function Base.getindex(data::DNSData{Ny, Nz}, start::Real, stop::Real; skip_step::Int=1) where {Ny, Nz}
    i_start = findfirst(x->x>=start, data.snaps); i_stop = findlast(x->x<=stop, data.snaps)
    isnothing(i_start) || isnothing(i_stop) ? throw(SnapshotTimeError(start, stop)) : DNSData{Ny, Nz, length(data.snaps[i_start:skip_step:i_stop])}(data.loc, data.params, data.snaps[i_start:skip_step:i_stop])
end
Base.firstindex(data::DNSData) = tryparse(Float64, data.snaps[firstindex(data.snaps)])
Base.lastindex(data::DNSData) = tryparse(Float64, data.snaps[lastindex(data.snaps)])

_read_params(loc::String) = read(Inifile(), loc)
_fetch_param(ini::Inifile, param::Symbol, ::Type{T}=Float64) where {T} = tryparse(T, strip(get(ini, "params", string(param)), ';'))
_getparamfield(data::DNSData, field::Symbol) = _fetch_param(getfield(data, :params), field)
_filterDirectoryToSnapshots(path) = filter!(x -> x=="params" || x=="K" || x=="t" ? false : true, readdir(path))
