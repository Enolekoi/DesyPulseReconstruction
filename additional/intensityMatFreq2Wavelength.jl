const clight = 299_792_458    # Lichtgeschwindigkeit
const c2p = 2*pi*clight

function intensityMatFreq2Wavelength(ww, freqIntMat)
    @assert length(ww) == size(freqIntMat, 2)
    @assert issorted(ww)
    wvIntMat = zeros(eltype(freqIntMat), size(freqIntMat))


    ll =  c2p ./ reverse(ww)
    llGrid = range(ll[1], ll[end], length(ll))
    @assert issorted(ll)

    for (i, Sw) in enumerate(eachrow(freqIntMat))
        Sl = reverse(Sw .* ww.^2 / c2p)       # (2.17) bei Trebino
        itp = linear_interpolation(ll, Sl)
        wvIntMat[i,:] = itp.(llGrid)
    end

    return (llGrid, wvIntMat)
end
