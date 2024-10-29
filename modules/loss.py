'''
loss.py Module

Module containing functions used for loss functions
'''
#############
## Imports ##
#############
import copy
import torch
import torch.nn as nn 
import torch.fft as trafo 
import matplotlib.pyplot as plt

from modules import config
from modules import data
from modules import helper
from modules import constants as c

'''
PulseRetrievalLossFunction()

Description:
    Loss function for pulse retrieval
'''
class PulseRetrievalLossFunction(nn.Module):
    def __init__(
            self, 
            pulse_threshold = 0.01,
            penalty = 10.0,
            real_weight = 1.0,
            imag_weight = 1.0,
            intensity_weight = 1.0,
            phase_weight = 1.0,
            frog_error_weight= 1.0
            ):
        '''
        Inputs:
            pulse_threshold     -> [float] threshold which determines minimum value considered to be a pulse
            real_weight         -> [float] weight used for MSE of real values
            imag_weight         -> [float] weight used for MSE of imaginary values
            intensity_weight    -> [float] weight used for MSE of intensity values
            phase_weight        -> [float] weight used for MSE of phase values
            frog_error_weight   -> [float] weight used for FROG-Error
        '''
        super(PulseRetrievalLossFunction, self).__init__() 
        self.pulse_threshold = pulse_threshold
        self.penalty = penalty
        self.real_weight = real_weight
        self.imag_weight = imag_weight
        self.intensity_weight = intensity_weight
        self.phase_weight = phase_weight
        self.frog_error_weight = frog_error_weight
        self.mse_weight_sum = real_weight + imag_weight + intensity_weight + phase_weight
         
        self.shg_transform = data.ResampleSHGmatrix(
            config.OUTPUT_NUM_DELAYS, 
            config.OUTPUT_TIMESTEP, 
            config.OUTPUT_NUM_FREQUENCIES,
            config.OUTPUT_START_FREQUENCY,
            config.OUTPUT_END_FREQUENCY,
            )

    def forward(self, prediction, label, shg_matrix, header):
        '''
        Inputs:
            prediction      -> [tensor] real part of predicted signal (imaginary part will be calculated)
            label           -> [tensor] real and imaginary part of the signal
            shg_matrix      -> [tensor] label shg_matrix for calculating FROG-Error
        Outputs:
            loss            -> [float] loss
        '''
        device = shg_matrix.device
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

        # Loop over each batch
        for i in range(batch_size):
            '''
            Calculate FROG Error
            '''
            if self.frog_error_weight != 0.0:
                # just get current index from batch
                original_shg = shg_matrix[i]
                original_header = header[i]
                # get original SHG-matrix (without 3 identical channels)
                original_shg = original_shg[0]
                # get a wavelength SHG-Matrix from the analytical signal and the header
                predicted_shg, new_header = createSHGmatFromAnalytical(
                        analytical_signal= prediction_analytical[i],
                        header=original_header
                        )
                # resample to correct size
                predicted_shg_data = [predicted_shg, new_header]
                _, prediction_header, predicted_shg, _, _ = self.shg_transform(predicted_shg_data)
                
                predicted_shg = predicted_shg[0,:,:]
                predicted_shg = helper.normalizeSHGmatrix(predicted_shg)
                
                # calculate_frog_error
                frog_error = calcFrogError(
                        t_ref  = original_shg, 
                        t_meas = predicted_shg
                        )

            
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
                first_significant_idx = min(first_significant_idx_real, first_significant_idx_imag) # determine the higher last significant index 
                last_significant_idx = max(last_significant_idx_real, last_significant_idx_imag)
                # create the phase mask
                phase_mask = torch.zeros(half_size).to(device)
                phase_mask[first_significant_idx:last_significant_idx] = 1
                # create the intensity_mse mask
                pulse_mask = torch.ones(half_size).to(device)
                pulse_mask[first_significant_idx:last_significant_idx] = self.penalty

                # Calculate MSE for the real and imaginary part
                mse_real = (prediction_real[i] - label_real[i]) ** 2
                mse_real = mse_real * pulse_mask
                mse_imag = (prediction_imag[i] - label_imag[i]) ** 2
                mse_imag = mse_imag * pulse_mask

                # Calculate MSE for the intensity and phase
                mse_intensity = (prediction_intensity[i] - label_intensity[i]) ** 2
                mse_intensity = mse_intensity * pulse_mask
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
'''
createSHGmat()

Description:
    Create a SHG-matrix from the analytical signal
Inputs:
    analytical_signal   -> [tensor] analytical time signal
    delta_tau           -> [float] time step between delays
    wCenter             -> [float] angular center frequency
Outputs:
    shg_matrix          -> [tensor] SHG-matrix
'''
def createSHGmat(analytical_signal, delta_tau, wCenter):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = len(analytical_signal)

    # create a tensor storing indicies starting with -N/2 to N/2
    start = -N // 2
    end = N // 2    
    delay_index_vector = torch.arange(start, end, dtype=torch.float32).to(device)

    # calculate shift factor
    shift_factor = torch.exp(-1j * 2 * wCenter * delta_tau * delay_index_vector).to(device)
    
    # initialize empty SHG-matrix
    shg_matrix = torch.zeros((N, N), dtype=torch.complex128).to(device)
    
    def fftshift(x):
        return torch.fft.fftshift(x)
    
    def circshift(x, shift):
        shift = int( shift % x.size(0)) 
        return torch.roll(x, shifts=shift, dims=0)

    for (matrix_index, delay_index) in enumerate(delay_index_vector):
        analytical_shifted = circshift(analytical_signal, delay_index).to(device)
        multiplied_matrixes = analytical_signal.to(device) * analytical_shifted * shift_factor
        fft_analytical = torch.fft.fft(fftshift(multiplied_matrixes))
        shg_matrix[matrix_index, :] = delta_tau * fftshift(fft_analytical)

    return shg_matrix

'''
createSHGmatFromAnalytic

Description:
    Create a SHG-matrix from an analytical signal and a header
Inputs:
    analytical_signal   -> [tensor] analytical signal
    header              -> [tensor] header containing information of a SHG-matrix
Outputs:
    shg_matrix          -> [tensor] SHG-matrix
    new_header          -> [tensor] header of the newly created SHG-matrix
'''
def createSHGmatFromAnalytical(analytical_signal, header):
    # get information from header
    num_delays      = header[0]
    num_wavelength  = header[1]
    delta_tau       = header[2]
    
    assert num_delays == num_wavelength
    
    N = num_delays

    # create temporary angular frequency axis for determining the center frequency
    temp_freq_axis = helper.frequencyAxisFromHeader(header)

    # get the angular center frequency from the temporary frequency axis
    center_frequency = helper.getCenterOfAxis(temp_freq_axis)

    # calculate SHG-Matrix from analytical signal
    shg_matrix_freq = createSHGmat(analytical_signal, delta_tau, center_frequency / 2)
    # get the intensity SHG-Matrix
    shg_matrix_freq = torch.abs(shg_matrix_freq) ** 2
    # normalize the SHG-matrix to [0, 1] 
    shg_matrix_freq = helper.normalizeSHGmatrix(shg_matrix_freq)

    # calculate angular frequency step between samples
    delta_nu = 1 / (N * delta_tau) 
    delta_omega = 2 * c.pi * delta_nu
    
    # get frequency axis 
    frequency_axis = helper.generateAxis(N=num_wavelength, resolution=delta_omega, center=center_frequency)

    # convert to wavelength
    wavelength_axis, shg_matrix = helper.intensityMatrixFreq2Wavelength(frequency_axis, shg_matrix_freq)
    # get new center_wavelength
    new_center_wavelength = helper.getCenterOfAxis(wavelength_axis)
    # calculate wavelength step size between samples
    new_delta_lambda = float(wavelength_axis[1] - wavelength_axis[0])

    # get shape of the new SHG-matrix
    new_num_delays = shg_matrix.size(0)
    new_num_wavelength = shg_matrix.size(1)

    # create the header for the newly created shg-matrix
    new_header = [
        new_num_delays,
        new_num_wavelength,
        delta_tau,
        new_delta_lambda,
        new_center_wavelength
        ]

    return shg_matrix, new_header

'''
calcFrogError()

Description:
    Calculate the FROG-Error out of two FROG-Traces
Inputs:
    t_ref       -> [tensor] reference FROG-Trace
    t_meas      -> [tensor] FROG-Trace to compare
Outputs:
    frog_error  -> [float] FROG-Error
'''
def calcFrogError(t_ref, t_meas):
    device = t_ref.device
    t_meas.to(device)
    # print(f"Max value of Tmeas = {torch.max(t_meas)}")
    # print(t_meas.shape)
    M, N = t_meas.shape
    # print(f"M = {M}")
    # print(f"N = {N}")
    # print(f"Tmeas = {t_meas}")
    sum1 = torch.sum(t_meas* t_ref)
    # print(f"Tmeas * Tref = {sum1}")
    sum2 = torch.sum(t_ref* t_ref)
    # print(f"Tref * Tref =  {sum2}")
    mu = sum1 / sum2 # pypret gl. 13 (s. 497)
    # print(f"mu = {mu}")
    # print(f"Tmeas-mu*Tref = {t_meas - mu*t_ref}")
    r = torch.sum(t_meas - mu*t_ref)**2    # pypret gl. 11 (s. 497) 
    # print(f"r = {r}")
    if(r != 0.0):
        normFactor = M * N * torch.max(t_meas)**2    # pypret gl. 12 (s. 497)
        # print(f"norm factor = {normFactor}")
        frog_error = torch.sqrt(r / normFactor)     # pypret gl. 12 (s. 497)
    else:
        frog_error = 0.0
    # print(f"FROG Error = {frog_error}")
    return frog_error

'''
hilbert()

Description:
    Calculate the hilbert transform of a real-valued signal
Inputs:
    signal              -> [tensor] real-valued signal [tensor]
    plot                -> [bool] if true plots of the hilber transform process are made
Outputs:
    analytical_signal   -> [tensor] the analytical signal (with the hilbert transform as the imaginary part) [tensor]
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
