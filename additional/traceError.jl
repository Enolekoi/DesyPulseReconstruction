# Hilfsfunktionen zur Berechnung des Trace-Errors
function mseUpToScale(A::AbstractArray, B::AbstractArray)
    mu = sum(A .* B) / sum(B.*B)   # Gl. (13) im pypret-Paper
    return sum( (A .- mu * B).^2 )     # Gl. (11) im pypret-Paper
end


function traceError(Tmeas::Matrix{T}, Tref::Matrix{T})::T where {T <: Real}
    r = mseUpToScale(Tmeas, Tref)
    normFactor = prod(size(Tmeas)) * maximum(Tmeas)^2   
    return sqrt(r/normFactor)   # Gl. (12) im pypret-Paper
end
