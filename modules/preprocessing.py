'''
Preprocessing
'''
#############
## Imports ##
#############
import logging
import torch
import numpy as np

from modules import constants as c
from modules import helper

'''
zeroCrossings()

Description:
    Get the zero crossings of tensor y
Inputs:
    x           -> [tensor] x-values of the tensor
    y           -> [tensor] y-values of the tensor
    tollerance  -> [float] absolute y-values smaller than this will be considered 0
Outputs:
    x_zero      -> [tensor] x-values where y = 0
'''
def zeroCrossings(x, y, tollerance=1e-12):

    N = len(x)
    x_zero = []
    # check if y and x have the same length
    if len(y) != N:
        raise ValueError("arguments x and y need to have the same length")

    # check if first element of y is close to zero (smaller than +-tollerance)
    if abs(y[0]) < tollerance:
        # add value to zero crossings
        x_zero.append(x[0])
    
    for n in range(1, N):
        # check if element is close to zero (smaller than +-tollerance)
        if abs(y[n]) < tollerance:
            # add value to zero crossings
            x_zero.append(x[n])
            continue    # go to next index n
        
        # check for changing sign between y[n] and y[n-1] 
        # this is the case when y[n] * y[n-1] is negative and y[n-1] isn't inside the tollerance considered zero
        if y[n] * y[n-1] < 0 and abs(y[n-1]) > tollerance:
            # calculate the difference between x[n] and x[n-1]
            delta_x = x[n] - x[n-1]
            if not (delta_x > 0):
                raise ValueError("Difference between x-values needs to be > 0")

            # calculate the difference between y[n] and y[n-1]
            delta_y = y[n] - y[n-1]

            # calculate the slope
            slope = delta_y / delta_x
            assert abs(slope) > 0 # check that slope is not 0
            
            # calculate intercepts
            intercept_1 = y[n-1] - slope * x[n-1]
            intercept_2 = y[n] - slope * x[n]
            # mean intercept
            mean_intercept = (intercept_1 + intercept_2) / 2
            
            # calculate zero
            zero = -mean_intercept / slope
            
            x_zero.append(zero)

    return x_zero

'''
calcFWHM()

Description:
    Calculate Full Width Half Maximum value
Inputs:
    yd      -> [tensor] Signal of which the FWHM is to be calculated
    tt      -> [tensor] Axis of the signal
Outputs:
    fwhm    -> [float] full width half maximum
'''
def calcFWHM(yd, tt):
    # find the maximum value and its index of yd
    x_m = np.max(yd)
    # index_m = np.argmax(yd)

    # substract the half of the maximum from yd
    yd_half_maximum = yd - x_m / 2
    # get the zero crossings relative to half its maximum
    xz = zeroCrossings(tt, yd_half_maximum)

    # if there are less than 2 zero crossings, return -1 and print a message
    if len(xz) < 2:
        print("FWHM cannot be calculated, since there are fewer than 2 half-maximum points")
        return -1
    else:
        # calculate the full width half maximum as the difference between the highest and lowest zero crossing
        fwhm = np.max(xz) - np.min(xz)
        return fwhm


'''
preprocessRawShgMatrix()

Description:
    Preprocess experimental SHG-matrixes
Inputs:

Outputs:
'''
def preprocessRawShgMatrix(spectrogrm_matrix, header, nTarget):
    # TODO: Window function to minimize trails at the maxima of the time axis
    # extract header information
    num_delays          = header[0] # number of delay samples
    num_wavelength      = header[1] # number of wavelength samples
    time_step           = header[2] # time step between delays [s]
    wavelength_step     = header[3] # distance between wavelength samples [m]
    center_wavelength   = header[4] # center wavelength in [m]

    # if nTarget is not even
    if nTarget % 2 !=0:
        raise ValueError("nTarget must be even")

    # if the shape of the spectrogrm_matrix is not even
    if spectrogrm_matrix.shape[0] != num_delays or spectrogrm_matrix.shape[1] != num_wavelength:
        raise ValueError("spectrogram matrix and header information don't match!")

    # 1: Symmetrically trim around the center of mass in the delay direction
    # get the sum of all spectrogram values
    total_int = float(torch.sum(spectrogrm_matrix))

    com_delay_index = 0     # index of center of mass in delay direction
    # enumerate over rows
    for index, row in enumerate(spectrogrm_matrix):
        # get sum of row
        sum_row = torch.sum(row)
        # add previous center of mas index and the current index multiplied by the sum of the current row
        com_delay_index += float((index + 1) * sum_row )
    
    com_delay_index = round(com_delay_index / total_int)
    distance_to_end = min(num_delays - com_delay_index, com_delay_index)
    index_range = torch.arange(com_delay_index - distance_to_end, com_delay_index + distance_to_end + 1)
    
    assert index_range[0] >= 0
    assert index_range[-1] < num_delays

    # Cut a matrix that is equally as large in both directions of com_delay_index
    # This means com_delay_index is exactly in the middle
    symmetric_spectrogram_matrix = spectrogrm_matrix[index_range, :]
    num_symmetric_delays = symmetric_spectrogram_matrix.shape[0]    # size of matrix in delay direction (uneven by design)
    assert num_symmetric_delays % 2 == 0

    # Construct Delay Axis for symmetric_spectrogram_matrix
    middle_index = (num_symmetric_delays + 1) // 2  # since num_symmetric_delays is uneven (see above)
    symmetric_delay_axis = (torch.arange(num_symmetric_delays) - (middle_index - 1)) * time_step
    assert abs(symmetric_delay_axis[middle_index - 1]) < time_step / 1e6    # check that the delay at the center index is 0

    # 2: Symmetrization of matrix around the center of mass by getting the mean left and right halves
    left_matrix = symmetric_spectrogram_matrix[:middle_index-1, :]
    right_matrix = symmetric_spectrogram_matrix[middle_index:, :]
    # get the mean of left and mirrored right half of the matrix
    left_symmetric = 0.5 * (left_matrix + torch.flip(right_matrix, dims=[0]))
    # replace left and right half by their mean and the flipped mean
    symmetric_spectrogram_matrix[:middle_index-1, :] = left_symmetric
    symmetric_spectrogram_matrix[middle_index:, :] = torch.flip(left_symmetric, dims=[0])

    # 3: Resampling with tau and lambda estimation
    # estimate full width half mean in both directions (not perfectly exact, but close enough)
    mean_delay_profile = torch.mean(symmetric_spectrogram_matrix, dim=1)
    fwhm_delay = calcFWHM(mean_delay_profile, symmetric_delay_axis)
    # if calcFWHM throws an error:
    if fwhm_delay < 0.0:
        raise ValueError("fwhm_delay could not be calculated")

    # get the full width half mean of the wavelengths
    # get the wavelength axis
    _, wavelenght_axis = helper.generateAxes(header)
    mean_spectrogram_profile = torch.mean(symmetric_spectrogram_matrix, dim=1)
    fwhm_wavelength = calcFWHM(mean_spectrogram_profile, wavelenght_axis)
    # if calcFWHM throws an error:
    if fwhm_wavelength < 0.0:
        raise ValueError("fwhm_wavelength could not be calculated")

    # Trebino Formula 10.8
    M = np.sqrt(fwhm_delay * fwhm_wavelength * nTarget * c.c0 / center_wavelength**2)
    opt_delay = fwhm_delay / M
    opt_wavelength = fwhm_wavelength / M

    # construction of axes for resampling
    index_vector = torch.arange(-nTarget // 2, nTarget // 2)
    resampled_delay_axis = index_vector * opt_delay
    resampled_wavelength_axis = index_vector * opt_wavelength + center_wavelength

    # get the index range of the new (resampled) axes, that lay between the axes
    # of the original matrix. Done to prevent extrapolating, which fails with noise
    # TODO:
    idxRangeNewDelay = idx_range_within_limits(resampledDelayAxis, (symmDelayAxis.min(), symmDelayAxis.max()))

    # get noise from first and last three columns of the original spectrogram    
    noise = torch.cat((spectrogrm_matrix[:, :3].flatten(), spectrogrm_matrix[:, -3:].flatten()))
    resampled_matrix = torch.std(noise) * torch.randn(nTarget, nTarget) + torch.mean(noise)
