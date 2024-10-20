module DNSAnalyse

using IniFile, FFTW, Printf, Mmap

using ReSolverInterface

export DNSData, loadDNS
export dns2field!, dns2field, correct_mean!
export mean!, mean

include("snapshoterror.jl")
include("snapshot.jl")
include("dnsdata.jl")
include("mean.jl")
include("spectralanalysis.jl")

end
