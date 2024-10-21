module DNSAnalyse

using IniFile, FFTW, Printf, Mmap

export DNSData
export dns2field!, dns2field, correct_mean!
export mean!, mean

include("snapshoterror.jl")
include("snapshot.jl")
include("dnsdata.jl")
include("mean.jl")
include("spectralanalysis.jl")

end
