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

from torch.fft import fftshift as fftshift
from torch.fft import fft as fft

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
            use_label = True,
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
        self.use_label = use_label
        self.pulse_threshold = pulse_threshold
        self.penalty = penalty
        self.real_weight = real_weight
        self.imag_weight = imag_weight
        self.intensity_weight = intensity_weight
        self.phase_weight = phase_weight
        self.frog_error_weight = frog_error_weight
        self.mse_weight_sum = real_weight + imag_weight + intensity_weight + phase_weight
        self.remove_ambiguities = data.RemoveAmbiguitiesFromLabel(config.OUTPUT_SIZE)
        self.shg_resample = data.ResampleSHGmatrix(
            config.OUTPUT_NUM_DELAYS, 
            config.OUTPUT_TIMESTEP, 
            config.OUTPUT_NUM_FREQUENCIES,
            config.OUTPUT_START_FREQUENCY,
            config.OUTPUT_END_FREQUENCY,
            )
        self.shg_create3channels = data.Create3ChannelSHGmatrix()

    def forward(self, prediction, shg_matrix, header, label=None):
        '''
        Inputs:
            prediction      -> [tensor] real part of predicted signal (imaginary part will be calculated)
            label           -> [tensor] real and imaginary part of the signal
            shg_matrix      -> [tensor] label shg_matrix for calculating FROG-Error
        Outputs:
            loss            -> [float] loss
        '''
        device = shg_matrix.device
        batch_size, half_size = prediction.shape
        # create the analytical signal using the hilbert transformation
        prediction_analytical = torch.zeros(batch_size, half_size, dtype=torch.complex64)

        for i in range(batch_size):
            prediction_analytical[i] = hilbert(prediction[i], plot=False).to(device)

        # get real and imaginary parts of predictions
        prediction_real = prediction_analytical.real.to(device)
        prediction_imag = prediction_analytical.imag.to(device)
        # remove the ambiguities and get corrected prediction real and imaginary parts
        prediction_tensor = torch.cat([prediction_real, prediction_imag],dim=1)
        prediction_tensor = self.remove_ambiguities(prediction_tensor)
        prediction_real = prediction_tensor[:, :half_size].to(device)
        prediction_imag = prediction_tensor[:, half_size:].to(device)


        if self.use_label:
            # get real and imaginary parts of labels
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
                resample_outputs = self.shg_resample(predicted_shg_data)
                resample_outputs = self.shg_create3channels(resample_outputs)
                _, prediction_header, predicted_shg, _, _ = resample_outputs
                
                # get only one channel
                predicted_shg = predicted_shg[0,:,:]
                # normalize SHG-matrix
                predicted_shg = helper.normalizeSHGmatrix(predicted_shg)
                original_shg = helper.normalizeSHGmatrix(original_shg)
                
                # calculate_frog_error
                frog_error = calcFrogError(original_shg, predicted_shg)
            
            '''
            Weighted MSE-Error
            '''
            if self.use_label:

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
    Create an amplitude SHG-matrix from the analytical signal with delay and frequency axis.
    For the more common intensity SHG-matrix, take the elementwise squared absolute.
    Because of frequency doubling due to the SHG crystal the product signal is shifted by 2*wCenter
Inputs:
    E                   -> [tensor] analytical time signal
    delta_tau           -> [float] time step between delays / sampling time
    wCenter             -> [float] angular center frequency
Outputs:
    shg_matrix          -> [tensor] SHG-matrix
'''
def createSHGmat(analytical_signal, delta_tau, wCenter):
    # get correct device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get the amount of samples of the analytical_signal and rename it
    N = len(analytical_signal)
    E = analytical_signal.to(device)

    # double the frequency because of the SHG-matrix
    wCenter2 = 2 * wCenter

    # create a tensor storing indicies -N/2 to N/2
    start = -N // 2
    end = N // 2    
    delay_index_tensor = torch.arange(start, end, dtype=torch.float32).to(device)

    # create time tensor (t)
    time_axis = delta_tau * delay_index_tensor

    # calculate shift factor e^(-j 2*\omega_c * t)
    shift_factor = torch.exp(-1j * wCenter2 * time_axis).to(device)
    
    # initialize empty SHG-matrix
    shg_matrix = torch.zeros((N, N), dtype=torch.complex128).to(device)
    
    # increment over the 
    for (matrix_index, delay_index) in enumerate(delay_index_tensor):
        # Shift E-Field for the current delay index
        E_shifted = helper.circshift(E, delay_index)
        # Calculate the the argument of the fft E(t)*E(t-\tau)*shift_factor
        argument = E * E_shifted * shift_factor
        # current matrix index is the shifted fft of E(t)*E(t-\tau)*shift_factor
        shg_matrix[matrix_index, :] = fftshift(fft(argument))

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
    # convert new header to tensor
    new_header = torch.tensor(new_header)
    # normalize SHG-matrix
    shg_matrix = helper.normalizeSHGmatrix(shg_matrix)

    return shg_matrix, new_header

'''
calcFrogError()

Description:
    Calculate the FROG-Error out of two FROG-Traces
Inputs:
    I_k         -> [tensor] retrieved FROG-Trace to compare
    I_m         -> [tensor] measured FROG-Trace to compare to
Outputs:
    frog_error  -> [float] FROG-Error
'''
def calcFrogError(I_k, I_m):
    # get correct device
    device = I_k.device
    # ensure all tensors are on the same device
    I_m.to(device)
    # get the shape of the measured spectrogram
    M, N = I_m.shape
    
    # calculate \mu [pypret gl. 13 (s. 497)]
    mu_numerator = torch.sum(I_m* I_k)
    print(f"mu_numerator = {mu_numerator}")
    mu_denominator = torch.sum(I_k* I_k)
    print(f"mu_denominator = {mu_denominator}")
    mu = mu_numerator / mu_denominator 
    print(f"mu = {mu}")
    
    # calculate normalization factor
    norm_factor = 1 / (M * N)
    print(f"norm_factor = {norm_factor}")
    # calculate the magnitude squared difference between the spectrograms
    mag_squared_difference = torch.abs(I_m - mu*I_k)**2
    print(f"mag_squared_difference = {mag_squared_difference}")
    # calculate the FROG-Error
    frog_error = torch.sqrt(norm_factor * torch.sum(mag_squared_difference))
    print(f"frog_error = {frog_error}")

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
    N = signal.size(0)  # Length of the input signal
    signal_fft = trafo.fft(signal)  # FFT of input signal
    freq_axis = trafo.fftfreq(N)  # FFT of input signal
    
    # Create the frequency mask for the hilbert transform
    H = torch.zeros(N, dtype=torch.complex64, device=signal.device)

    # if N is even
    if N % 2 == 0:
        N_half = N // 2     # Half of the signal (when N is even)
        H[0] = 1            # DC component
        H[N_half+1:] = 2    # Positive frequencies
        H[N_half] = 1       # Nyquist frequency (only for even N)
    else:
        N_half = (N+1) // 2 # Half of the signal (when N is uneven)
        H[0] = 1            # DC component
        H[:N_half] = 2    # Positive frequencies
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
        plt.plot(freq_axis, signal_fft, label='real part')
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
        plt.plot(freq_axis, signal_fft_hilbert, label='Signal after Hilbert Mask')
        plt.title('Signal after multiplication with the mask of the Hilbert transform')
        plt.ylabel('Intensity')
        plt.xlabel('Frequency')
        plt.legend()
        plt.grid()
        
        plt.show()
    
    return analytical_signal

def findMinimumFrogError(tensor1, tensor2):
    width, height = tensor1.shape

    min_error = float('inf')
    best_shift = 0
    best_shifted_tensor = None

    for shift in range(0, height):  # Allow shifting up and down
        shifted_tensor = torch.roll(tensor2, shifts=shift, dims=1)  # Circular shift along the vertical axis
        error = calcFrogError(shifted_tensor, tensor1)

        # Update the best shift if the error is lower
        if abs(error) < min_error:
            min_error = error
            best_shift = shift
            best_shifted_tensor = shifted_tensor

    return min_error, best_shifted_tensor

