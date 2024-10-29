'''
Preprocessing
'''
#############
## Imports ##
#############
import logging
import torch
import numpy as np
from scipy.interpolate import RectBivariateSpline as RectBivariateSpline
from scipy.signal import windows

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from modules import constants as c
from modules import data
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
    x_m = np.max(yd.numpy())
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
preprocessFromPath()
'''
def preprocessFromPath(path):
    N = 256
    reader = data.ReadSHGmatrix()
    shg_data = reader(path)
    shg_matrix, header = shg_data
    preprocessRawShgMatrix(shg_matrix, header, N)


'''
preprocessRawShgMatrix()

Description:
    Preprocess experimental SHG-matrixes
Inputs:

Outputs:
'''
def preprocessRawShgMatrix(spectrogrm_matrix, header, nTarget):
    # extract header information
    num_delays          = header[0] # number of delay samples
    num_wavelength      = header[1] # number of wavelength samples
    delta_tau           = header[2] # time step between delays [s]
    delta_lambda        = header[3] # distance between wavelength samples [m]
    center_wavelength   = header[4] # center wavelength in [m]
    spectrogrm_matrix = helper.normalizeSHGmatrix(spectrogrm_matrix)

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
        # add previous center of mass index and the current index
        # then multiply by the sum of the values in the current row
        com_delay_index += float((index + 1) * sum_row )
    
    # device the com_delay_index by the sum of total SHG-matrix values and round it 
    com_delay_index = round(com_delay_index / total_int)
    # get the minimum of either:
    #   - the number of delays - the center of mass index
    #   - the center of mass index
    # This will get the index which has the smallest distance to the end of the matrix
    distance_to_end = int(min(num_delays - com_delay_index - 1, com_delay_index))
    # Create a tensor with the index range to be in the new spectrogram
    # The smalles value will be the Center of Mass index - the smallest distance to the end of the original matrix 
    # The hightest value will be the Center of Mass index + the smallest distance to the end of the original matrix 
    # This means com_delay_index is exactly in the middle of the index range
    index_range = torch.arange(com_delay_index - distance_to_end, com_delay_index + distance_to_end + 1)
    
    assert index_range[0] >= 0
    assert index_range[-1] < num_delays

    # Cut a matrix that is equally as large in both directions of com_delay_index
    # This means com_delay_index is exactly in the middle
    symmetric_spectrogram_matrix = spectrogrm_matrix[index_range, :]
    num_symmetric_delays = symmetric_spectrogram_matrix.shape[0]    # size of matrix in delay direction (uneven by design)
    assert num_symmetric_delays % 2 == 1

    # Construct Delay Axis for symmetric_spectrogram_matrix
    middle_index = (num_symmetric_delays + 1) // 2  # since num_symmetric_delays is uneven (see above)
    # create the delay axis
    # get a range of values as long as the number of delays
    # shift it with around the middle index
    # scale it with delta_tau
    symmetric_delay_axis = (torch.arange(num_symmetric_delays) - (middle_index - 1)) * delta_tau
    assert abs(symmetric_delay_axis[middle_index - 1]) < delta_tau / 1e6    # check that the delay at the center index is 0

    # 2: Symmetrization of matrix around the center of mass by getting the mean left and right halves
    left_matrix = symmetric_spectrogram_matrix[:middle_index-1, :]
    right_matrix = symmetric_spectrogram_matrix[middle_index:, :]
    # get the mean of left and mirrored right half of the matrix
    left_symmetric = 0.5 * (left_matrix + torch.flip(right_matrix, dims=[0]))
    # replace left and right half by their mean and the flipped mean
    symmetric_spectrogram_matrix[:middle_index-1, :] = left_symmetric
    symmetric_spectrogram_matrix[middle_index:, :] = torch.flip(left_symmetric, dims=[0])
    print(symmetric_spectrogram_matrix.shape)

    # 3: Resampling with tau and lambda estimation
    # estimate full width half mean in both directions (not perfectly exact, but close enough)
    # get the mean the delay
    mean_delay_profile = torch.mean(symmetric_spectrogram_matrix, dim=1)
    print(len(mean_delay_profile))
    print(len(symmetric_delay_axis))
    # get the full width half mean of the delay
    fwhm_delay = calcFWHM(mean_delay_profile, symmetric_delay_axis)
    # if calcFWHM throws an error:
    if fwhm_delay < 0.0:
        raise ValueError("fwhm_delay could not be calculated")

    # get the wavelength axis
    _, wavelenght_axis = helper.generateAxes(header)
    # get the mean the wavelengths
    mean_spectrogram_profile = torch.mean(symmetric_spectrogram_matrix, dim=0)
    print(len(mean_spectrogram_profile))
    print(len(wavelenght_axis))
    # get the full width half mean of the wavelengths
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

    # get the index range, where the values of the resampled delay axis are within the 
    # limits of the symmetric delay axis
    # This is done to prevent extrapolating the SHG-matrix, since this fails with noise
    index_range_new_delay = indexRangeWithinLimits(
            resampled_delay_axis,
            (symmetric_delay_axis.min(), symmetric_delay_axis.max())
            )
    # get the subset of 'resampled_delay_axis', that is within the 'index_range_new_delay'
    resampled_delay_axis_subset = resampled_delay_axis[
            index_range_new_delay[0]:index_range_new_delay[1]
            ]

    # get the index range, where the values of the resampled wavelength axis are within the 
    # limits of the wavelength axis
    index_range_new_wavelength = indexRangeWithinLimits(
            resampled_wavelength_axis,
            (wavelenght_axis.min(), wavelenght_axis.max())
            )
    # get the subset of 'resampled_wavelength_axis', that is within the 'index_range_new_wavelength'
    resampled_wavelength_axis_subset = resampled_wavelength_axis[
            index_range_new_wavelength[0]:index_range_new_wavelength[1]
            ]
    
    # get noise from first and last three columns of the original spectrogram    
    noise = torch.cat((spectrogrm_matrix[:, :3].flatten(), spectrogrm_matrix[:, -3:].flatten()))
    resampled_shg_matrix = torch.std(noise) * torch.randn(nTarget, nTarget) + torch.mean(noise)

    # Windowing function to reduce the trails at the end of the original SHG-matrix
    # There would be a hard limit between the original SHG-matrix and the background
    # define hamming window
    window_width = int(symmetric_spectrogram_matrix.size(1))
    print(window_width)
    window_delay = torch.from_numpy( windows.gaussian(window_width, std=1000 )).float()
    # window_wavelength = torch.from_numpy(np.hanning( symmetric_spectrogram_matrix.size(1) )).float()
    # apply the window to the symmetric_spectrogram_matrix
    # 'None' adds a new axis for broadcasting
    windowed_spectrogram = symmetric_spectrogram_matrix * window_delay[None, :]
    # windowed_spectrogram = windowed_spectrogram * window_wavelength[None, :]
    
    # 2D interpolate
    # initialize the interpolator
    interpolator = RectBivariateSpline(
            symmetric_delay_axis.numpy(),
            wavelenght_axis.numpy(),
            windowed_spectrogram.numpy()
            )
    # perform interpolation
    shg_interpolated = interpolator(
            resampled_delay_axis_subset.numpy(),
            resampled_wavelength_axis_subset.numpy()
            ).T
    index_delay_range_min = index_range_new_delay[0]
    index_delay_range_max = index_range_new_delay[1] 
    index_wavelength_range_min = index_range_new_wavelength[0]
    index_wavelength_range_max = index_range_new_wavelength[1]
    # Embed the interpolated matrix into the larger matrix
    resampled_shg_matrix = resampled_shg_matrix.numpy()
    resampled_shg_matrix[index_wavelength_range_min:index_wavelength_range_max,\
            index_delay_range_min:index_delay_range_max] = shg_interpolated
    resampled_shg_matrix = torch.from_numpy(resampled_shg_matrix)

    # new header
    delta_tau = float(resampled_delay_axis[1] - resampled_delay_axis[0])
    delta_lambda = float(resampled_wavelength_axis[1] - resampled_wavelength_axis[0])
    center_wavelength = helper.getCenterOfAxis(resampled_wavelength_axis)
    num_delays = int(resampled_delay_axis.size(0))
    num_wavelength = int(resampled_wavelength_axis.size(0))
    
    new_header = [
            num_delays,
            num_wavelength,
            delta_tau,
            delta_lambda,
            center_wavelength
            ]
    print(f"New Header = {new_header}")
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    # fix the axes to a specific exponent representation
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-15,-15))    # use 1e-15 as exponent for x axis
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9))      # use 1e-9  as exponent for y axis
    # Plot the spectrogram
    plt.pcolormesh(
            resampled_delay_axis.numpy(),
            resampled_wavelength_axis.numpy(),
            resampled_shg_matrix.numpy(),
            shading='auto',
            # norm=LogNorm(vmin=1e-10, vmax=float( resampled_shg_matrix.max() ))
            )

    # Add labels and title
    plt.colorbar(label='Intensity')
    plt.ylabel("Wavelength [m]")
    plt.xlabel("Time [s]")
    plt.title("SHG Matrix")

    # Show the plot
    plt.show()

    return resampled_shg_matrix

'''
indexRangeWithinLimits()

Description:
    get the index range for which values of signal are within limits
Inputs:
    signal          -> [tensor] input signal
    limits          -> [list] limits in which the signal should fit
Outputs:
    index_range     -> [list] Index range where values of signal are within the limits

'''
def indexRangeWithinLimits(signal, limits):
    assert torch.all(signal.diff() >=0), "Tensor must be sorted in ascending order"
    assert limits[1] > limits [0], "Upper limit must be greater then lower limit"

    # find the first index where signal is greater than the lower limit
    greater_than_lower_limit = signal > limits[0]
    if greater_than_lower_limit.any():
        index_1 = greater_than_lower_limit.nonzero(as_tuple=True)[0].min().item()
    else:
        index_1 = None

    # find the last index where signal is greater than the lower limit
    less_than_upper_limit = signal < limits[1]
    if less_than_upper_limit.any():
        index_2 = less_than_upper_limit.nonzero(as_tuple=True)[0].max().item()
    else:
        index_2 = None

    index_range = (index_1, index_2)

    return index_range
