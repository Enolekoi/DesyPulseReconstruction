'''
loss.py Module

Module containing functions used for loss functions
'''

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.fft as trafo 
import config
import helper

import matplotlib.pyplot as plt

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.penalty_factor = penalty_factor
        self.threshold = threshold
        self.spec_transform = helper.ResampleSpectrogram(
            config.OUTPUT_NUM_DELAYS, 
            config.OUTPUT_TIMESTEP, 
            config.OUTPUT_NUM_FREQUENCIES,
            config.OUTPUT_START_FREQUENCY,
            config.OUTPUT_END_FREQUENCY,
            )

    def forward(self, prediction, label, spectrogram):
        device = spectrogram.device
        # print(f"prediction size = {predictions}")
        # print(f"label size = {labels}")
        
        # get the number of batches, as well as the shape of the labels
        batch_size, num_elements = label.shape
        # get half of elements
        half_size = num_elements // 2
        # get real and imaginary parts of labels and predictions
        label_real = label[:, :half_size].to(device)
        label_imag = label[:, half_size:].to(device)

        prediction_real = prediction[:, :half_size].to(device)
        prediction_imag = prediction[:, half_size:].to(device)

        label_phase =      torch.atan2(label_imag, label_real)
        prediction_phase = torch.atan2(prediction_imag, prediction_real)

        label_intensity = label_real**2 + label_imag**2
        prediction_intensity = prediction_real**2 + prediction_imag**2

        label_analytical = torch.complex(label_real, label_imag).to(device)
        prediction_analytical = torch.complex(prediction_real, prediction_imag).to(device)

        # initialize loss
        loss = 0.0

        # Loop over each batch
        for i in range(batch_size):
            '''
            MSE Error
            '''
            phase_mask = abs(label_intensity[i]) < 0.01

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
            mse_phase[phase_mask] = 0
            # Add to total loss
            loss += mse_real.mean() + mse_imag.mean() + 10*mse_intensity.mean() + 5*mse_phase.mean()
        # devide by batch size 
        loss = loss / batch_size

        return loss

'''
PulseRetrievalLossFunction():
Loss function for pulse retrieval
'''
class PulseRetrievalLossFunctionHilbertFrog(nn.Module):
    def __init__(
            self, 
            pulse_threshold = 0.01,
            real_weight = 1.0,
            imag_weight = 1.0,
            intensity_weight = 1.0,
            phase_weight = 1.0,
            frog_error_weight= 1.0
            ):
        '''
        Initialization
        Inputs:
            pulse_threshold     -> threshold which determines minimum value considered to be a pulse [float]
            real_weight         -> weight used for MSE of real values [float]
            imag_weight         -> weight used for MSE of imaginary values [float]
            intensity_weight    -> weight used for MSE of intensity values [float]
            phase_weight        -> weight used for MSE of phase values [float]
            frog_error_weight   -> weight used for FROG-Error [float] 
            
        '''
        super(PulseRetrievalLossFunctionHilbertFrog, self).__init__()
        self.pulse_threshold = pulse_threshold
        self.real_weight = real_weight
        self.imag_weight = imag_weight
        self.intensity_weight = intensity_weight
        self.phase_weight = phase_weight
        self.frog_error_weight = frog_error_weight
        self.mse_weight_sum = real_weight + imag_weight + intensity_weight + phase_weight
         
        self.spec_transform = helper.ResampleSpectrogram(
            config.OUTPUT_NUM_DELAYS, 
            config.OUTPUT_TIMESTEP, 
            config.OUTPUT_NUM_FREQUENCIES,
            config.OUTPUT_START_FREQUENCY,
            config.OUTPUT_END_FREQUENCY,
            )
        self.c0 = 299792458

    def forward(self, prediction, label, spectrogram, header):
        '''
        Inputs:
            prediction      -> real part of predicted signal (imaginary part will be calculated) [tensor]
            label           -> real and imaginary part of the signal [tensor]
            spectrogram     -> label spectrogram for calculating FROG-Error
        Outputs:
            loss            -> loss
        '''
        device = spectrogram.device
        spec_shape = spectrogram.shape
        # get the number of batches, as well as the shape of the labels
        batch_size, num_elements = label.shape
        # get half of elements
        half_size = num_elements // 2

        # create the analytical signal using the hilbert transformation
        prediction_analytical = torch.zeros(batch_size, half_size, dtype=torch.complex64)
        for i in range(batch_size):
            prediction_analytical[i] = hilbert(prediction[i], plot=False).to(device)

        # get real and imaginary parts of labels and predictions
        prediction_real = prediction_analytical.real.to(device)
        prediction_imag = prediction_analytical.imag.to(device)
        label_real = label[:, :half_size].to(device)
        label_imag = label[:, half_size:].to(device)

        # calculate intensities
        label_intensity = label_real**2 + label_imag**2
        prediction_intensity = prediction_real**2 + prediction_imag**2

        # calculate phases
        label_phase = torch.atan2(label_imag, label_real)
        prediction_phase = torch.atan2(prediction_imag, prediction_real)

        # initialize losses
        loss = 0.0
        mse_loss = 0.0
        frog_error = 0.0
        # initialize some values for SHG-trace creation
        prediction_header = header

        # Loop over each batch
        for i in range(batch_size):
            '''
            Calculate FROG Error
            '''
            if self.frog_error_weight != 0.0:
                # just get current index from batch
                original_spectrogram = spectrogram[i]
                # get original spectrogram (without 3 identical channels)
                original_spectrogram = original_spectrogram[0]
                
                # create new SHG Matrix
                predicted_spectrogram = createSHGmat(
                        yta = prediction_analytical[i],
                        Ts = prediction_header[i][2],
                        wCenter = 2*torch.pi * self.c0 / prediction_header[i][4]
                        )

                # get FROG intensity from FROG amplitude
                predicted_spectrogram = (torch.abs(predicted_spectrogram)**2).to(device)
                predicted_spectrogram = helper.min_max_normalize_spectrogram(spectrogram)

                predicted_spectrogram_data = [predicted_spectrogram, header]
                in_spectrogram, prediction_header, predicted_spectrogram, output_time, output_freq = self.spec_transform(predicted_spectrogram_data)

                # print(f"1 = {torch.max(predicted_spectrogram)}")
                # calculate_frog_error
                frog_error = calcFrogError(
                        Tref  = original_spectrogram, 
                        Tmeas = predicted_spectrogram
                        )
                print(f"FROG Error: {frog_error}")
            
            '''
            Weighted MSE-Error
            '''
            if self.mse_weight_sum != 0.0:

                # Create masks for all absolute values higher than the threshold
                mask_real_threshold = abs(label_real[i]) > self.pulse_threshold
                mask_imag_threshold = abs(label_imag[i]) > self.pulse_threshold
            
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

                # determine the lower first significant index
                first_significant_idx = min(first_significant_idx_real, first_significant_idx_imag)
                # determine the higher last significant index
                last_significant_idx = max(last_significant_idx_real, last_significant_idx_imag)
                # create the phase mask
                phase_mask = torch.zeros(half_size)
                phase_mask[first_significant_idx:last_significant_idx] = 1

                # Calculate MSE for the real and imaginary part
                mse_real = (prediction_real[i] - label_real[i]) ** 2
                mse_imag = (prediction_imag[i] - label_imag[i]) ** 2
                
                # Calculate MSE for the intensity and phase
                mse_intensity = (prediction_intensity[i] - label_intensity[i]) ** 2
                mse_phase = (prediction_phase[i] - label_phase[i]) ** 2 
                # Use the phase mask for phase blanking
                mse_phase = mse_phase * phase_mask

                # calculate weighted means for MSE
                mse_real_mean = mse_real.mean()*self.real_weight
                mse_imag_mean = mse_imag.mean()*self.imag_weight
                mse_intensity_mean = mse_intensity.mean()*self.intensity_weight
                mse_phase_mean = mse_phase.mean()*self.phase_weight
                # calculate weighted mse loss
                mse_loss += (mse_real_mean + mse_imag_mean + mse_intensity_mean + mse_phase_mean) / self.mse_weight_sum

            # calculate weighted mean loss of mse loss and frog error
            loss += mse_loss + frog_error*self.frog_error_weight
        # devide by batch size 
        loss = loss / batch_size

        return loss


def getCenterFreq(yta, time_step):
    device = yta.device
    n = yta.size(0)
    # Calculate the fft of the analytical signal
    yta_fft = trafo.fft(yta)
    # Calculate the frequencies that correspond to the fourier coefficients (frequency bins)
    frequencies = trafo.fftfreq(n, d=time_step).to(device)
    print(frequencies)

    # Calculate the power spectrum
    power_spectrum = (torch.abs(yta_fft)**2).to(device)

    positive_frequencies = frequencies[frequencies >= 0]
    positive_power_spectrum = power_spectrum[frequencies >= 0]

    # Calculate center frequency (weighted average of the frequency bins)
    wCenter = torch.sum(positive_frequencies * positive_power_spectrum) / torch.sum(positive_power_spectrum)
    return 2*torch.pi *wCenter


def createSHGmat(yta, Ts, wCenter):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = len(yta)

    # create a tensor storing indicies starting with -N/2 to N/2
    start = -N // 2
    end = N // 2    
    delayIdxVec = torch.arange(start, end, dtype=torch.float32).to(device)

    # calculate shift factor
    # argument0 = 2*wCenter
    # argument1 = -1j * 2 * wCenter * Ts
    # argument2 = argument1 * delayIdxVec
    # print(f"2*wCenter = {argument0}")
    # print(f"-j 2*wCenter*Ts = {argument1}")
    # print(f"-j 2*wCenter*Ts*delayIdxVec = {argument2}")

    # calculate shift factor
    shiftFactor = torch.exp(-1j * 2 * wCenter * Ts * delayIdxVec).to(device)
    # print(f"Shift factor = {shiftFactor}")

    shgMat = torch.zeros((N, N), dtype=torch.complex64)

    def fftshift(x):
        return torch.fft.fftshift(x)
    
    def circshift(x, shift):
        shift = int( shift % x.size(0)) 
        return torch.roll(x, shifts=shift, dims=0)

    for (matIdx, delayIdx) in enumerate(delayIdxVec):
        ytaShifted = circshift(yta, delayIdx).to(device)
        multiplied_matrixes = yta * ytaShifted * shiftFactor
        fft_yta = torch.fft.fft(fftshift(multiplied_matrixes))
        shgMat[matIdx, :] = Ts * fftshift(fft_yta)
    return shgMat

def calcFrogError(Tref, Tmeas):
    device = Tref.device
    Tmeas.to(device)
    # print(f"Max value of Tref = {torch.max(Tref)}")
    # print(f"Max value of Tmeas = {torch.max(Tmeas)}")
    M, N = Tmeas.shape
    # print(f"M = {M}")
    # print(f"N = {N}")
    sum1 = torch.sum(Tmeas* Tref)
    # print(f"Tmeas * Tref = {sum1}")
    sum2 = torch.sum(Tref* Tref)
    # print(f"Tref * Tref =  {sum2}")
    mu = sum1 / sum2 # pypret gl. 13 (s. 497)
    # print(f"mu = {mu}")
    # print(f"Tmeas-mu*Tref = {Tmeas - mu*Tref}")
    r = torch.sum(Tmeas - mu*Tref)**2    # pypret gl. 11 (s. 497) 
    # print(f"r = {r}")
    if(r != 0.0):
        normFactor = M * N * torch.max(Tmeas)**2    # pypret gl. 12 (s. 497)
        # print(f"norm factor = {normFactor}")
        frog_error = torch.sqrt(r / normFactor)     # pypret gl. 12 (s. 497)
    else:
        frog_error = 0.0
    # print(f"FROG Error = {frog_error}")
    return frog_error

'''
hilbert()

Calculate the hilbert transforma of a real-valued signal
'''
def hilbert(signal, plot=False):
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
        # H[0] = 1            # DC component
        H[N_half+1:] = 2    # Positive frequencies
        H[N_half] = 1       # Nyquist frequency (only for even N)
    else:
        N_half = (N+1) // 2 # Half of the signal (when N is uneven)
        # H[0] = 1            # DC component
        H[N_half:] = 2    # Positive frequencies
    # apply the frequency mask
    signal_fft_hilbert = signal_fft * H

    # inverse FFT to get the analytical signal
    analytical_signal = trafo.ifft(signal_fft_hilbert)
    
    if plot:
        plt.figure()
        # plot 1 (time domain, real part)
        plt.subplot(4,1,1)
        plt.plot(signal, label='real part')
        plt.title('Real part of time domain signal')
        plt.ylabel('Intensity')
        plt.xlabel('Time')
        plt.legend()
        plt.grid()
        
        # plot 2 (frequency domain, real part[fft] )
        plt.subplot(4,1,2)
        plt.plot(signal_fft, label='real part')
        plt.title('FFT of the real part of time domain signal')
        plt.ylabel('Intensity')
        plt.xlabel('Frequency')
        plt.legend()
        plt.grid()
        
        # plot 3 (frequency domain, hilbert mask)
        plt.subplot(4,1,3)
        plt.plot(H, label='Hilbert Mask')
        plt.title('Mask of the Hilbert transform')
        plt.ylabel('Intensity')
        plt.xlabel('Frequency')
        plt.legend()
        plt.grid()
        
        # plot 4 (frequency domain, hilbert mask)
        plt.subplot(4,1,4)
        plt.plot(signal_fft_hilbert, label='Signal after Hilbert Mask')
        plt.title('Signal after multiplication with the mask of the Hilbert transform')
        plt.ylabel('Intensity')
        plt.xlabel('Frequency')
        plt.legend()
        plt.grid()
        
        plt.show()
    
    return analytical_signal
