"""
  createSHGmat: berechnet die SHG-Frog-Matrix aus dem analytischen Signal im Zeitbereich.
  Dabei wird angenommen, dass das Zeitsignal im Bereich [-T0/2, T0/2 - Ts] definiert ist (mit T0 = N*Ts).
  Wegen der Frequenzverdopplung wird das Produktsignal um 2*wCenter im Frequenzbereich verschoben.
  # Arguments
  `yta`: analytisches Signal im Zeitbereich
  `Ts`: Abtastzeit = Delay
  `wCenter`: Center-Frequenz
  # Return value: Eine quadratische, komplexe NxN-Matrix, wobei N die Laenge des Zeitvektors yta ist.
  # Das ist die 'Frog-Amplitude'. Für die übliche Frog-Trace (also Intensität) muss mann noch das elementweise Betragsquadrat bilden. 
"""
function createSHGmat(yta, Ts,  wCenter)
    N = length(yta)
    @assert iseven(N)
    delayIdxVec = (-N ÷ 2):(N ÷ 2 - 1)
    shiftFactor = exp.(- 1im * 2*wCenter * Ts * delayIdxVec)

    shgMat = zeros(Complex, N, N)
    for (matIdx, delayIdx) in enumerate(delayIdxVec)
        ytaShifted = circshift(yta, delayIdx)
        shgMat[matIdx, :] = Ts * fftshift(fft(fftshift(yta .* ytaShifted .* shiftFactor)))
    end
    return shgMat   # Frog-Amplitude
end 



