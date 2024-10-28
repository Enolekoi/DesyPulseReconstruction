'''
helper.py Module

Module containing various helper functions
'''
#############
## Imports ##
#############
import logging
import torch

from modules import constants as c
logger = logging.getLogger(__name__)

'''
RemoveAmbiguitiesFromLabel()
Read labels (real and imag part) from Es.dat
'''
class RemoveAmbiguitiesFromLabel(object):
    def __init__(self, number_elements):
        '''
        Inputs:
            number_elements -> [int] Number of elements in the intensity and phase array each
        '''
        self.number_elements = number_elements
        # get the center index of each half
        self.center_index = number_elements // 2

    def __call__(self, label):    
        '''
        Read Ek.dat file and place columns in arrays
        Inputs:
            label           -> [tensor] List of arrays containing real and imag part of time signal
        Outputs:
            output_label    -> [tensor] List of arrays containing real and imag part of time signal, without ambiguities
        '''
        real_part = label[:self.number_elements]
        imag_part = label[self.number_elements:]

        complex_signal = torch.complex(real_part, imag_part)

        # remove translation ambiguity Eamb(t) = E(t-t0)
        complex_signal = removeTranslationAmbiguity(complex_signal, self.center_index)
        # remove mirrored complex conjugate ambiguity Eamb(t) = E*(-t)
        complex_signal = removeConjugationAmbiguity(complex_signal, self.center_index)
        # remove absolute phase shift ambiguity Eamb(t) = E(t)*exp(j\phi0)
        complex_signal = removePhaseShiftAmbiguity(complex_signal, self.center_index)

        real = complex_signal.real
        imag = complex_signal.imag

        output_label = torch.cat([real, imag])
        return output_label

'''
removeTranslationAmbiguity()

Description:
    Remove the delay translation ambiguity from a complex signal
Inputs:
    complex_signal          -> [tensor] complex signal
    center_index            -> [int] center index of the signal tensor
Outputs:
    complex_signal_noambig  -> [tensor] complex signal without the delay translation ambiguity
'''
def removeTranslationAmbiguity(complex_signal, center_index):
    # calculate the intensity of the signal
    intensity = complex_signal.real**2 + complex_signal.imag**2

    # get the index of the highest intensity value and calculate the offset
    peak_index = torch.argmax(intensity)
    offset = center_index - peak_index

    # shift the real and imaginary parts to center the peak
    complex_signal_noambig = torch.roll(complex_signal, offset.item() )

    return complex_signal_noambig

'''
removeConjugationAmbiguity()

Description:
    Remove the complex conjugation and mirroring ambiguity from a complex signal
Inputs:
    complex_signal          -> [tensor] complex signal
    center_index            -> [int] center index of the signal tensor
Outputs:
    complex_signal_noambig  -> [tensor] complex signal without the complex conjugation and mirroring ambiguity
'''
def removeConjugationAmbiguity(complex_signal, center_index):
    # calculate the intensity of the signal
    intensity = complex_signal.real**2 + complex_signal.imag**2

    # Calculate the mean of the pulse before and after the center index
    mean_first_half = torch.mean(intensity[:center_index])
    mean_second_half = torch.mean(intensity[center_index:])
    
    # if the weight of the signal is in the second half of the signal
    if mean_second_half > mean_first_half:
        # conjugate the signal
        complex_signal_conjugated = complex_signal.conj()
        # mirror the signal
        complex_signal_noambig = torch.flip(complex_signal_conjugated, dims=[0])
    # if the weight of the signal is in the first half of the signal
    else:   
        # do nothing
        complex_signal_noambig = complex_signal

    return complex_signal_noambig

'''
removePhaseShiftAmbiguity()

Description:
    Remove the absolute phase shift ambiguity from a complex signal
Inputs:
    complex_signal          -> [tensor] complex signal
    center_index            -> [int] center index of the signal tensor
Outputs:
    complex_signal_noambig  -> [tensor] complex signal without the absolute phase shift ambiguity
'''
def removePhaseShiftAmbiguity(complex_signal, center_index):
    # calculate the phase of the signal [rad]
    phase = torch.angle(complex_signal)

    # get phase at center index
    center_phase = phase[center_index]
    # remove the absolute phase shift from the whole phase tensor
    phase = phase - center_phase

    # reconstruct real and imaginary parts
    # get the magnitude
    magnitude = torch.abs(complex_signal)
    # calculate real part
    real_part = magnitude * torch.cos(phase) 
    imag_part = magnitude * torch.sin(phase)
    
    # create a new complex tensor
    complex_signal_noambig = torch.complex(real_part, imag_part)
    
    return complex_signal_noambig

'''
frequencyAxisFromHeader()

Description:
    Create an equidistant frequency_axis based on the header of a SHG-Matrix
Inputs:
    header                      -> [tensor] 5 element header containing information about a SHG-Matrix
Outputs:
    equidistant_frequency_axis  -> [tensor] equidistant frequency axis
'''
def frequencyAxisFromHeader(header):
    # get header information
    _, \
    num_wavelength,\
    _, \
    delta_lambda, \
    center_wavelength = header
    
    # number of frequency samples are equal to number of wavelength samples
    num_frequency = num_wavelength

    # Create the wavelength axis
    wavelength_axis = generateAxis(
            N = num_wavelength,
            resolution = delta_lambda,
            center = center_wavelength
            )
    # convert to frequency axis which is not equidistant
    frequency_axis = c.c2pi / wavelength_axis
    # get extrema
    min_freq = frequency_axis.min()
    max_freq = frequency_axis.max()

    # create an equidistant frequency axis
    equidistant_frequency_axis = torch.linspace(min_freq, max_freq, steps=num_frequency)

    return equidistant_frequency_axis

'''
getCenterOfAxis()

Description:
    return the center element of an axis
Inputs:
    axis            -> [tensor] wavelength/delay/frequency axis
Outputs:
    center_element  -> [float] center element of the axis
'''
def getCenterOfAxis(axis):
    length_axis = axis.size(0)
    # calculate the center index
    center_index = length_axis // 2
    # get the center element
    center_element =  float(axis[center_index])
    return center_element

'''
normalizeSHGMatrix()

Description:
    normalize a SHG-matrix to the range [0, 1]
Inputs:
    shg_matrix              -> [tensor] SHG-matrix
Outputs:
    shg_matrix_normalized   -> [tensor] normalized SHG-matrix
'''
def normalizeSHGmatrix(shg_matrix):
    shg_min = shg_matrix.min()
    shg_max = shg_matrix.max()

    shg_matrix_normalized = (shg_matrix - shg_min) / (shg_max - shg_min)

    return shg_matrix_normalized

'''
batchPiecewiseLinearInterpolation()

Description:
    Interpolate signal
Inputs:
    x       -> [tensor] x-values of the signal
    y       -> [tensor] y-values of the signal
    x_new   -> [tensor] new x-values the y-Values are being interpolated to
Outputs:
    y_new   -> [tensor] interpolated y-values
'''
def batchPiecewiseLinearInterpolation(x, y, x_new):
    # Ensure inputs are at least 2D, adding a batch dimension if necessary
    if x.ndim == 1:
        x = x.unsqueeze(0)  # Add a batch dimension
    if y.ndim == 1:
        y = y.unsqueeze(0)  # Add a batch dimension
    if x_new.ndim == 1:
        x_new = x_new.unsqueeze(0)  # Add a batch dimension
    # Initialize y_new with the same shape as x_new to store interpolated results
    y_new = torch.zeros_like(x_new)

    # Loop over each segment in x and y
    for i in range(x.shape[-1] - 1):
        # Expand dimensions for broadcasting in batch mode
        x_i, x_next = x[:, i:i+1], x[:, i+1:i+2]
        y_i, y_next = y[:, i:i+1], y[:, i+1:i+2]

        # Calculate mask where x_new is within the current segment [x_i, x_next]
        mask = (x_new >= x_i) & (x_new <= x_next)

        if mask.any():
            # Calculate slopes (m) in batch
            delta_x = x_next - x_i
            delta_y = y_next - y_i
            m = delta_y / delta_x

            # Apply piecewise linear interpolation where mask is True
            y_new[mask] = (y_i + m * (x_new[mask] - x_i))[mask]

    return y_new

'''
piecewiseLinearInterpolation()

Description:
    Interpolate signal
Inputs:
    x       -> [tensor] x-values of the signal
    y       -> [tensor] y-values of the signal
    x_new   -> [tensor] new x-values the y-Values are being interpolated to
Outputs:
    y_new   -> [tensor] interpolated y-values
'''
def piecewiseLinearInterpolation(x, y, x_new):
    y_new = torch.zeros_like(x_new)

    indices = torch.searchsorted(x, x_new)
    indices = torch.clamp(indices, 1, len(x) -1)

    lower_indices = indices - 1
    upper_indices = indices

    x_lower = x[lower_indices]
    x_upper = x[upper_indices]
    y_lower = y[lower_indices]
    y_upper = y[upper_indices]

    delta_y = y_upper - y_lower
    delta_x = x_upper - x_lower
    m = delta_y / delta_x

    y_new = y_lower + m * (x_new - x_lower)

    return y_new

'''
generateAxis()

Description:
    Generates Axis based on given parameters
Inputs:
    N           -> [int] length of the axis
    resolution  -> [float] value between samples of the axis
    center      -> [float] value around which the axis is centered
Outputs:
    axis        -> [tensor] Axis
'''
def generateAxis(N, resolution, center=0.0):
    # generate indicies
    if N % 2 == 0:
        start = -(N // 2)
        end = (N // 2) - 1
    else:
        start = -(N//2)
        end = N // 2 
    # generate indices
    index = torch.arange(start, end + 1)
    
    # ensure the length is N
    assert len(index) == N

    # create axis by scaling indices with the resolution and adding center
    axis = index * resolution + center

    return axis

''' 
generateAxes()

Description:
    Generate time and wavelength Axes from the header of a SHG-matrix
Inputs:
    header              -> [list] header of a SHG-matrix
Outputs:
    delay_axis          -> [tensor] delay axis
    wavelength_axis     -> [tensor] wavelength axis
'''
def generateAxes(header):
    # extract header information
    num_delays          = header[0] # number of delay samples
    num_wavelength      = header[1] # number of wavelength samples
    delta_tau           = header[2] # time step between delays [s]
    delta_lambda        = header[3] # distance between wavelength samples [m]
    center_wavelength   = header[4] # center wavelength in [m]

    # create the delay axis
    delay_axis = generateAxis(
            N = num_delays, 
            resolution = delta_tau,
            center = 0.0
            )

     # create the wavelength axis
    wavelength_axis = generateAxis(
            N = num_wavelength, 
            resolution = delta_lambda,
            center = center_wavelength
            )

    # return axes
    return delay_axis, wavelength_axis

'''
intensityMatrixFreq2Wavelength()

Description:
    Convert an intensity SHG-matrix from frequency to wavelength
Inputs:
    frequency_axis                  -> [tensor] frequency axis of the intensity SHG-matrix
    freq_intensity_matrix           -> [tensor] intensity SHG-matrix (frequency)
Outputs:
    wavelength_axis_equidistant     -> [tensor] wavelength axis of the new intensity SHG-matrix
    wavelength_intensity_matrix     -> [tensor] intensity SHG-matrix (wavelength)
'''
def intensityMatrixFreq2Wavelength(frequency_axis, freq_intensity_matrix):
    length = len(frequency_axis)
    # print(f"length = {length}")
    assert length == freq_intensity_matrix.size(1)

    # initialize output matrix with zeros in the shape of the input matrix
    wavelength_intensity_matrix = torch.zeros_like(freq_intensity_matrix)

    # calculate the wavelength_axis and flip it for decending order
    wavelength_axis = c.c2pi / frequency_axis.flip(0) 
    # get the equidistant wavelength axis
    wavelength_min = wavelength_axis[0]
    wavelength_max = wavelength_axis[-1]
    wavelength_axis_equidistant = torch.linspace(wavelength_min, wavelength_max, length)
    
    # calculate wavelength intensity matrix (2.17 Trebino)
    # itterate over each row 
    for i, Sw in enumerate(freq_intensity_matrix):
        # Element wise operation (Sw .* wavelength_axis.^2 / c2p)
        Sl = torch.flip(Sw * (frequency_axis ** 2) / c.c2pi, dims=[0])
        # Reshape Sl to a 3D tensor for interpolation (batch size, channel_size, original length)
        # Sl = Sl.unsqueeze(0).unsqueeze(0)

        # perform linear interpolation
        interpolated = piecewiseLinearInterpolation(wavelength_axis, Sl, wavelength_axis_equidistant)

        # assign interpolated values to freq_intensity_matrix
        wavelength_intensity_matrix[i, :] = interpolated

    return wavelength_axis_equidistant, wavelength_intensity_matrix
