'''
loss.py Module

Module containing functions used for loss functions
'''

import torch
import torch.nn as nn 
import torch.fft as trafo 
import config
import helper

'''
PulseRetrievalLossFunction():
Loss function for pulse retrieval
'''
class PulseRetrievalLossFunction(nn.Module):
    def __init__(self, penalty_factor=2.0, threshold=0.01):
        '''
        Initialization
        Inputs:
            weight_factor   -> Factor by which the loss is multiplied, when the label is greater than the threshold [float]
            threshold       -> Label value over which the higher weights get multiplied with the loss [float]
        '''
        super(PulseRetrievalLossFunction, self).__init__()

        self.penalty_factor = penalty_factor
        self.threshold = threshold
        self.spec_transform = helper.ResampleSpectrogram(
            config.OUTPUT_NUM_DELAYS, 
            config.OUTPUT_TIMESTEP, 
            config.OUTPUT_NUM_FREQUENCIES,
            config.OUTPUT_START_FREQUENCY,
            config.OUTPUT_END_FREQUENCY
            )



    def forward(self, prediction, label, spectrogram):
        # print(f"prediction size = {predictions}")
        # print(f"label size = {labels}")
        
        # get the number of batches, as well as the shape of the labels
        batch_size, num_elements = label.shape
        # get half of elements
        half_size = num_elements // 2

        # get real and imaginary parts of labels and predictions
        label_real = label[:, :half_size]
        label_imag = label[:, half_size:]

        prediction_real = prediction[:, :half_size]
        prediction_imag = prediction[:, half_size:]

        label_phase =      torch.atan2(label_imag, label_real)
        prediction_phase = torch.atan2(prediction_imag, prediction_real)

        label_intensity = label_real**2 + label_imag**2
        prediction_intensity = prediction_real**2 + prediction_imag**2

        label_analytical = torch.complex(label_real, label_imag)
        prediction_analytical = torch.complex(prediction_real, prediction_imag)

        # initialize loss
        loss = 0.0
        frog_error = 0.0
        Ts = 1.5
        wCenter = 337.927

        # Loop over each batch
        for i in range(batch_size):

            # create new SHG Matrix
            predicted_spectrogram = createSHGmat(prediction_analytical, Ts, wCenter)
            # resample to correct size
            predicted_spectrogram = self.spec_transform(predicted_spectrogram)
            # calculate_frog_error
            frog_error = calcFrogError(predicted_spectrogram, spectrogram)

            # Create masks for all absolute values higher than the threshold
            mask_real_threshold = abs(label_real[i]) > self.threshold
            mask_imag_threshold = abs(label_imag[i]) > self.threshold
            
            # if any real value is greater than the threshold
            if torch.any(mask_real_threshold):
                # get the first and last index, where a value is greater than the threshold
                first_significant_idx_real = torch.nonzero(mask_real_threshold).min().item()
                last_significant_idx_real = torch.nonzero(mask_real_threshold).max().item()
            else:
                first_significant_idx_real = 0
                last_significant_idx_real = half_size - 1

            # if any imaginary value is greater than the threshold
            if torch.any(mask_imag_threshold):
                # get the first and last index, where a value is greater than the threshold
                first_significant_idx_imag = torch.nonzero(mask_imag_threshold).min().item()
                last_significant_idx_imag = torch.nonzero(mask_imag_threshold).max().item()
            else:
                first_significant_idx_imag = 0
                last_significant_idx_imag = half_size - 1

            # Calculate MSE for the real and imaginary part
            mse_real = (prediction_real[i] - label_real[i]) ** 2
            mse_imag = (prediction_imag[i] - label_imag[i]) ** 2

            # Apply penalty for values before the first significant index and after the last
            mse_real[:first_significant_idx_real] *= self.penalty_factor
            mse_real[last_significant_idx_real + 1:] *= self.penalty_factor
            mse_imag[:first_significant_idx_imag] *= self.penalty_factor
            mse_imag[last_significant_idx_imag + 1:] *= self.penalty_factor
            
            mse_intensity = (prediction_intensity[i] - label_intensity[i]) ** 2
            mse_phase = (prediction_phase[i] - label_phase[i]) ** 2

            # Add to total loss
            # loss += mse_real.mean() + mse_imag.mean() + 5*mse_intensity.mean() + 0*mse_phase.mean()
            loss += frog_error
        # devide by batch size 
        loss = loss / batch_size

        return loss

def createSHGmat(yta, Ts, wCenter):
    N = len(yta)
    # create a tensor storing indicies starting with -N/2 to N/2
    start = -N // 2
    end = N // 2    
    delayIdxVec = torch.arange(start, end, dtype=torch.float32)

    # calculate shift factor
    shiftFactor = torch.exp(-1j * 2 * wCenter * Ts * delayIdxVec)

    shgMat = torch.zeros((N, N), dtype=torch.complex64)

    def fftshift(x):
        return torch.fft.fftshift(x)
    
    def circshift(x, shift):
        shift = shift % x.size(0)
        return torch.roll(x, shifts=shift, dims=0)

    for (matIdx, delayIdx) in enumerate(delayIdxVec):
        ytaShifted = circshift(yta, delayIdx)
        fft_yta = torch.fft.fft(fftshift(yta* ytaShifted * shiftFactor))
        shgMat[matIdx, :] = Ts * fftshift(fft_yta)
    return shgMat

def calcFrogError(Tref, Tmeas):

    mu = torch.sum(torch.matmul(Tmeas, Tref)) / torch.sum(torch.matmul(Tref, Tref)) # pypret gl. 13 (s. 497)
    r = torch.sum( (Tmeas - mu*Tref)**2)    # pypret gl. 11 (s. 497)

    normFactor = torch.prod(torch.tensor(Tmeas.shape)) * torch.max(Tmeas)**2    # pypret gl. 12 (s. 497)
    frog_error = torch.sqrt(r / normFactor)     # pypret gl. 12 (s. 497)
    return frog_error

'''
hilbert()

Calculate the hilbert transforma of a real-valued signal
'''
def hilbert(signal):
    """
    Inputs:
        signal  -> real-valued signal [tensor]
    Outputs:
        analytical_signal -> the analytical signal (with the hilbert transform as the imaginary part) [tensor]
    """
    N = signal.size(0)  # Length of the input signal
    signal_fft = trafo.fft(signal)  # FFT of input signal
    
    # Create the frequency mask for the hilbert transform
    H = torch.zeros(N, dtype=torch.complex64, device=signal.device)

    # if N is even
    if N % 2 == 0:
        N_half = N // 2     # Half of the signal (when N is even)
        H[0] = 1            # DC component
        H[1: N_half] = 2    # Positive frequencies
        H[N_half] = 1       # Nyquist frequency (only for even N)
    else:
        N_half = (N+1) // 2 # Half of the signal (when N is uneven)
        H[0] = 1            # DC component
        H[1: N_half] = 2    # Positive frequencies

    # apply the frequency mask
    signal_fft_hilbert = signal_fft * H

    # inverse FFT to get the analytical signal
    analytical_signal = trafo.ifft(signal_fft_hilbert)

    return analytical_signal
