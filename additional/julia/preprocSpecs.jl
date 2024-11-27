# Precossing des ganzen Verzeichnisses
using CairoMakie
CairoMakie.activate!(visible=false)
# include("/home/kjuen/dvlp/kai/Julia/PulseDataGeneration/pdg/PDG.jl")

#* Setup
N = 256
specDir = "/home/kjuen/dvlp/kai/FrogDaten/Data_with_the_Correct_Description/spectrograms"
preProcDir = "/home/kjuen/dvlp/kai/FrogDaten/Data_with_the_Correct_Description/preproc" * string(N)

specFiles = filter(f->occursin(r"^spectrogram.*.txt$", f), readdir(specDir))
# specNums = parse.(Int, map(f -> split(splitext(f)[1], "_")[2], specFiles))

#* Hilfsfunktionen
"""
  zeroCrossings: berechnet die Nulldurchgänge eines Vektors y
"""
function zeroCrossings(x::AbstractVector,
                       y::AbstractVector;
                       TOL = 1e-12)::Vector
    N = length(x)
    if length(y) != N
        error("Argumente müssen die gleiche Länge haben")
    end

    xzero = []

    if abs(y[1]) < TOL
        push!(xzero, x[1])
    end

    for n=2:N
        if abs(y[n]) < TOL
            push!(xzero, x[n])
            continue
        end
        if y[n]*y[n-1] < 0 && abs(y[n-1]) > TOL
            # fit a straight line bewteen the points and find the zero of this line
            Dx = x[n] - x[n-1]
            if !(Dx>0)
                error("Abstand der x-Werte muss > 0 sein")
            end
            Dy = y[n] - y[n-1]
            m = Dy / Dx
            @assert abs(m) > 0
            b1 = y[n-1] - m * x[n-1]
            b2 = y[n] - m * x[n]
            b = (b1 + b2) / 2
            push!(xzero, -b/m)
        end
    end
    return xzero
end


function berechneFWHM(yd::Vector{<:Real}, tt::AbstractVector)::Real
    (xm, im) = findmax(yd)
    xz = zeroCrossings(tt, yd .- xm/2)
    if length(xz) < 2
        println("FHWM kann nicht berechnet werden, da <2 HW-Punkte")
        # error("FHWM kann nicht berechnet werden, da <2 HW-Punkte")
        return -1.0
    else # mehr als 2
        ext = extrema(xz)
        return ext[2] - ext[1]
    end
end


struct SpectrogramInfo
    NumDelays::Int64          # Number of delays (rows)
    NumSpecPoints::Int64      # Number of points in spectrum (columns)
    DelayDist::Float64        # Distance between delays
    ResSpec::Float64          # Resolution of spectrum
    CenterWavelen::Float64    # Center Wavelength
    function SpectrogramInfo(headerString::AbstractString)
        items = collect(eachsplit(headerString))
        if length(items) != 5
            error("Headerstring besteht nicht aus 5 Werten")
        end
        new(parse(Int64, items[1]),
            parse(Int64, items[2]),
            parse(Float64, items[3])*femto,  # given in fs
            parse(Float64, items[4])*nano,   # given in nm
            parse(Float64, items[5])*nano)   # given in nm
    end
end



function generateAxis(N, res, center=0.0)
    idx = ceil(Int, -N/2) : 1 : floor(Int, (N-1)/2)
    @assert length(idx) == N
    return idx * res .+ center
end

function generateAxes(hi::SpectrogramInfo)
    delayAxis = generateAxis(hi.NumDelays, hi.DelayDist)
    @assert delayAxis[1] < delayAxis[end]
    wavAxis = generateAxis(hi.NumSpecPoints, hi.ResSpec, hi.CenterWavelen)
     #@assert wavAxis[1] < wavAxis[end]
    return (delayAxis, wavAxis)
end



#* Die eigentliche Preprocessing-Funktion


"""
 Preprocessing einer SHG-Spektrogramm-Matrix 'specMat'.
 hi sind die Informationen aus der Header-Zeile und nTarget ist die Größe der Rückgabe-Matrix

"""
function preprocRawSHGmat(specMat::Matrix{<: Real}, hi::SpectrogramInfo,
                          nTarget::Integer)

    if !iseven(nTarget)
        error("nTarget muss gerade sein")
    end


    if (size(specMat, 1) != hi.NumDelays) || (size(specMat, 2) != hi.NumSpecPoints)
        error("specMat und hi passen nicht zusammen")
    end

    #1: symmetrisch um den Schwerpunkt in delay-Richtung zuschneiden
    totalInt = sum(specMat)

    comDelayIdx = 0     # Index des Schwerpunktes (center of mass) in Delay-Richtung
    for (idx, row) in enumerate(eachrow(specMat))
        sr = sum(row)
        comDelayIdx += idx * sr
    end
    comDelayIdx = round(Integer, comDelayIdx / totalInt)
    distToEnd = min(hi.NumDelays - comDelayIdx, comDelayIdx)
    idxRange = (comDelayIdx - distToEnd) : (comDelayIdx + distToEnd)
    @assert idxRange[1] >= 1
    @assert idxRange[end] <= hi.NumDelays
    # hier wird eine Matrix ausgeschnitten, die vom comDelayIdx gleich groß in beide Richtungen ist.
    # Das heißt, der comDelayIdx ist definitiv in der Mitte.
    symmSpecMat = specMat[idxRange,:]
    NumSymmDelays = size(symmSpecMat, 1)   # Größe der Matrix in Delay-Richtung, ungerade per Konstruktion
    @assert isodd(size(symmSpecMat, 1))

    # Konstruktion der Delay-Achse für symmSpecMat
    middleIdx = (NumSymmDelays + 1) ÷ 2   # NumSymmDelays ist ja ungerade...
    symmDelayAxis = ((1:NumSymmDelays) .- middleIdx) * hi.DelayDist
    @assert abs(symmDelayAxis[middleIdx]) < hi.DelayDist / 1e6    # Der Delay am Mittelindex = 0

    #2: Symmetrisierung um Schwerpunkt in delay-Richtung durch Mittelwertbildung
    # von linkem und rechtem Teil der Matrix
    leftMat = symmSpecMat[1:(middleIdx-1),:]
    rightMat = symmSpecMat[(middleIdx+1):end, :]
    leftSymm = 1/2 * (leftMat .+ reverse(rightMat; dims=1))
    symmSpecMat[1:(middleIdx-1),:] = leftSymm
    symmSpecMat[(middleIdx+1):end, :] = reverse(leftSymm; dims=1)

    #3: Resampling mit passendem tau und lambda

    # Schaetzung von fhwm in beide Richtungen: das ist nicht ganz exakt, kommt aber einigermassen hin...
    meanDelayProfile = map(mean, eachrow(symmSpecMat))
    fwhmDelay = berechneFWHM(meanDelayProfile, symmDelayAxis) / sqrt(2)
    if fwhmDelay < 0.0
        error("fwhmDelay konnte nicht berechnet werden")
    end

    wavAxis = generateAxes(hi)[2]
    meanSpecProfile = map(mean, eachcol(symmSpecMat))
    fwhmWv = berechneFWHM(meanSpecProfile, wavAxis)#  *2* sqrt(2)
    if fwhmWv < 0.0
        error("fwhmWv konnte nicht berechnet werden")
    end

    # Formel 10.8 bei Trebino
    M = sqrt(fwhmDelay * fwhmWv * nTarget * clight / hi.CenterWavelen^2)
    optDelay = fwhmDelay / M
    optWvRes = fwhmWv / M

    # Konstruktion der Achsen fuer das Resampling
    idxVec = ((-nTarget ÷ 2):(nTarget ÷ 2 - 1))
    resampledDelayAxis = idxVec * optDelay
    resampledWvAxis = idxVec * optWvRes .+ hi.CenterWavelen

    # Bestimmung der Indexbereiche der neuen (geresamplten) Achsen, die innerhalb der
    # Achsen der Originalmatrix liegen. Damit vermeidet man das Extrapolieren,
    # was bei dem Rauschen nicht klappt
    idxRangeNewDelay = idxRangeWithinLimits(resampledDelayAxis, extrema(symmDelayAxis))
    resampledDelayAxisSubset = resampledDelayAxis[idxRangeNewDelay[1]:idxRangeNewDelay[2]]
    idxRangeNewWv = idxRangeWithinLimits(resampledWvAxis, extrema(wavAxis))
    resampledWvAxisSubset = resampledWvAxis[idxRangeNewWv[1]:idxRangeNewWv[2]]

    # Rauschen aus den ersten und letzten drei Spalten der Originalmatrix:
    noise = [vec(specMat[:,1:3]); vec(specMat[:, (end-2):end])]
    resampledMat = std(noise) * randn(nTarget, nTarget) .+ mean(noise)

    # 2D-Interpolation
    shgInterp = zeros(length(resampledDelayAxisSubset), length(resampledWvAxisSubset))
    itpc = cubic_spline_interpolation((symmDelayAxis, wavAxis), symmSpecMat)
    shgInterp = @. itpc(resampledDelayAxisSubset, resampledWvAxisSubset')
    # Die interpolierte Matrix in die Gesamt-Matrix einbetten:
    resampledMat[idxRangeNewDelay[1]:idxRangeNewDelay[2], idxRangeNewWv[1]:idxRangeNewWv[2]] = shgInterp

    return (resampledMat, resampledDelayAxis, resampledWvAxis)
end
