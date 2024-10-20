# Definitions for the snapshot struct and its interface

struct Snapshot{Ny, Nz}
    loc::String
    t::Float64
    K::Float64
    dKdt::Float64
    U::Array{Float64, 3}
end

function Snapshot(loc::String, Ny::Int, Nz::Int)
    # extract metadata
    (_, t, K, dKdt) = open(loc*"metadata") do f; tryparse.(Float64, lstrip.(x->isnothing(tryparse(Int, string(x))), readlines(f))); end
    t = round.(t; digits=6); K = round.(K; digits=6); dKdt = round.(dKdt; digits=6)

    # extract velocity field
    U = zeros(Float64, 3, Ny, Nz)
    open(loc*"U") do f; permutedims!(U, mmap(f, Array{Float64, 3}, (Nz + 1, Ny, 3))[1:end - 1, :, :], (3, 2, 1)); end

    Snapshot{Ny, Nz}(loc, t, K, dKdt, U)
end

function Base.getproperty(snap::Snapshot{Ny, Nz}, field::Symbol) where {Ny, Nz}
    if field === :omega
        omega = zeros(Float64, Ny, Nz)
        open(snap.loc*"omega") do f; omega .= mmap(f, Matrix{Float64}, (Ny, Nz)); end
        return omega
    elseif field === :psi
        psi = zeros(Float64, Ny, Nz)
        open(snap.loc*"psi") do f; psi .= mmap(f, Matrix{Float64}, (Ny, Nz)); end
        return psi
    else
        getfield(snap, field)
    end
end

Base.parent(snap::Snapshot) = snap.U

Base.getindex(snap::Snapshot, i::Int) = @view(snap.U[i, :, :])

Base.iterate(snap::Snapshot) = (snap[1], Val(:V))
Base.iterate(snap::Snapshot, ::Val{:V}) = (snap[2], Val(:W))
Base.iterate(snap::Snapshot, ::Val{:W}) = (snap[3], Val(:done))
Base.iterate(::Snapshot, ::Val{:done}) = nothing
