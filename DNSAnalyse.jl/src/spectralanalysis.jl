# This file contains a number of analysis helper functions, especially for spectral analysis of a
# series of data.

function unpackBinaryStream(path, ::Type{T}=Float64) where {T}
    arrayOutput = Vector{T}(undef, filesize(path)÷8)

    open(path, "r") do f
        read!(f, arrayOutput)
    end

    return arrayOutput
end

readKineticEnergy(path) = unpackBinaryStream.((path*"K", path*"t"))

hannWindow(t) = sin(π*t)^2
welchWindow(t) = 1 - (2t - 1)^2
sineWindow(t) = sin(π*t)
hammingWindow(t) = 0.53836 + 0.46164*cos(2π*t)

removeMean(timeSeries) = timeSeries .- Statistics.mean(timeSeries)
windowData(timeSeries, windowFunction) = map(x->windowFunction(x[1]/length(timeSeries))*x[2], enumerate(timeSeries))

function powerSpectra(timeSeries, sampleRate, windowFunction=hannWindow)
    timeSeriesFluctuations = removeMean(timeSeries)
    windowedTimeSeriesFluctuations = windowData(timeSeriesFluctuations, windowFunction)
    return abs2.(rfft(windowedTimeSeriesFluctuations)./length(timeSeries)), rfftfreq(length(timeSeries), sampleRate)
end

function filter_peaks(spectra, freqs, period)
    peak_indices, peaks = findmaxima(spectra)
    interpPeakFreq = zero(peaks)
    interpPeakValue = zero(peaks)
    for (i, peak_index) in enumerate(peak_indices)
        interpPeakFreq[i], interpPeakValue[i] = interpolatePeak(getInterpolationPoints(peak_index, spectra, freqs)..., period)
    end
    return interpPeakFreq, interpPeakValue
end

function spectraEnergy(spectra, energyThreshold=1.0)
    spectralEnergy = sum(spectra)
    energyFraction = 0.0
    i = 0
    for spectralComponent in spectra
        i += 1
        energyFraction += spectralComponent/spectralEnergy
        energyFraction > energyThreshold ? break : nothing
    end
    
    return i
end

function getInterpolationPoints(peakIndex, spectrum, freqs)
    return freqs[peakIndex], spectrum[[peakIndex - 1, peakIndex, peakIndex + 1]]
end

function interpolatePeak(freq, spectralPoints, dataPeriod)
    spectralPeakLocationBins = ((spectralPoints[1] - spectralPoints[3])/(2*(spectralPoints[1] - 2*spectralPoints[2] + spectralPoints[3])))
    spectralPeakMagnitude = spectralPoints[2] - (1/4)*(spectralPoints[1] - spectralPoints[3])*spectralPeakLocationBins
    return freq + (spectralPeakLocationBins/dataPeriod), spectralPeakMagnitude
end
