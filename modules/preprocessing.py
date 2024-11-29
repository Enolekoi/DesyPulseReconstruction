'''
Preprocessing
'''
#############
## Imports ##
#############
import os
import shutil
import logging
import torch
import numpy as np
import csv
from scipy.interpolate import RectBivariateSpline
from scipy.signal import windows

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from modules import constants as c
from modules import config
from modules import data
from modules import helper

logger = logging.getLogger(__name__)
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
        logger.info("FWHM cannot be calculated, since there are fewer than 2 half-maximum points")
        return -1
    else:
        # calculate the full width half maximum as the difference between the highest and lowest zero crossing
        fwhm = np.max(xz) - np.min(xz)
        return fwhm

'''
windowSHGmatrix()

Description:
    Use a window function on the SHG-matrix
Inputs:
    shg_matrix                  -> [tensor] SHG-matrix over which to apply the windowing function
    dimension                   -> [int] Dimension across which the window will be applied
    standard_deviation_factor   -> [float] the standard deviation of the gauss window equals window_width*standard_deviation_factor
Outputs:
    shg_matrix_windowed     -> [tensor] SHG-matrix after application of the windowing function
'''
def windowSHGmatrix(shg_matrix, dimension = 1, standard_deviation_factor = 0.1):
    # Raise exceptions
    if not( (dimension == 1) or (dimension == 0) ):
        raise ValueError(f"dimension should be either 1 or 0 but is {dimension}")
    if not (standard_deviation_factor > 0):
        raise ValueError(f"standard_deviation_factor needs to be higher than 0, but is {standard_deviation_factor}")
    shg_matrix_windowed = shg_matrix

    # determine the window width (equals the number of samples across the selected dimension)
    window_width = int(shg_matrix.size(dimension))
    # calculate the standard deviation used for the gaussian window
    window_standard_deviation = window_width * standard_deviation_factor
    logger.info(f"Window width                        = {window_width}")
    logger.info(f"Standard deviation of the window    = {window_standard_deviation}")
    # define the window
    window_delay = torch.from_numpy( windows.gaussian(window_width, std=window_standard_deviation)).float()

    # apply the window to the SHG-matrix
    # 'None' adds a new axis for broadcasting
    if dimension == 1:
        shg_matrix_windowed = shg_matrix * window_delay[None, :]
    elif dimension == 0:
        shg_matrix_windowed = shg_matrix * window_delay[:, None]

    return shg_matrix_windowed

'''
preprocessRawShgMatrix()

Description:
    Preprocess experimental SHG-matrixes
Inputs:

Outputs:
'''
def preprocessRawShgMatrix(shg_matrix, header, nTarget):
    # extract header information
    num_delays          = header[0] # number of delay samples
    num_wavelength      = header[1] # number of wavelength samples
    delta_tau           = header[2] # time step between delays [s]
    delta_lambda        = header[3] # distance between wavelength samples [m]
    center_wavelength   = header[4] # center wavelength in [m]
    # normalize SHG-Matrix
    shg_matrix = helper.normalizeSHGmatrix(shg_matrix)
    shg_matrix_original = shg_matrix

    # Windowing function to reduce the trails at the end of the original SHG-matrix
    # There would be a hard limit between the original SHG-matrix and the background
    # shg_matrix = windowSHGmatrix(
    #         shg_matrix = shg_matrix, 
    #         dimension = 0
    #         )
    # shg_matrix = windowSHGmatrix(
    #         shg_matrix = shg_matrix, 
    #         dimension = 1
    #         )

    # get the delay and wavelength axis
    delay_axis, wavelength_axis = helper.generateAxes(header)

    # if nTarget is not even
    if nTarget % 2 !=0:
        raise ValueError("nTarget must be even")

    # if the shape of the spectrogram_matrix is not even
    if shg_matrix.shape[0] != num_delays or shg_matrix.shape[1] != num_wavelength:
        raise ValueError("shg matrix and header information don't match!")

    # 1: Symmetrically trim around the center of mass in the delay direction
    # get the sum of all spectrogram values
    total_int = float(torch.sum(shg_matrix))

    com_delay_index = 0     # index of center of mass in delay direction
    # enumerate over rows
    for index, row in enumerate(shg_matrix):
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
    # Create a tensor with the index range to be in the new SHG-matrix
    # The smalles value will be the Center of Mass index - the smallest distance to the end of the original matrix 
    # The hightest value will be the Center of Mass index + the smallest distance to the end of the original matrix 
    # This means com_delay_index is exactly in the middle of the index range
    index_range = torch.arange(com_delay_index - distance_to_end, com_delay_index + distance_to_end + 1)
    
    assert index_range[0] >= 0
    assert index_range[-1] < num_delays

    # Cut a matrix that is equally as large in both directions of com_delay_index
    # This means com_delay_index is exactly in the middle
    symmetric_shg_matrix = shg_matrix[index_range, :]
    num_symmetric_delays = symmetric_shg_matrix.shape[0]    # size of matrix in delay direction (uneven by design)
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
    left_matrix = symmetric_shg_matrix[:middle_index-1, :]
    right_matrix = symmetric_shg_matrix[middle_index:, :]
    # get the mean of left and mirrored right half of the matrix
    left_symmetric = 0.5 * (left_matrix + torch.flip(right_matrix, dims=[0]))
    # replace left and right half by their mean and the flipped mean
    symmetric_shg_matrix[:middle_index-1, :] = left_symmetric
    symmetric_shg_matrix[middle_index:, :] = torch.flip(left_symmetric, dims=[0])

    # 3: Resampling with tau and lambda estimation
    # estimate full width half mean in both directions (not perfectly exact, but close enough)
    # get the mean the delay
    mean_delay_profile = torch.mean(symmetric_shg_matrix, dim=1)
    # get the full width half mean of the delay
    fwhm_delay = calcFWHM(mean_delay_profile, symmetric_delay_axis)
    # if calcFWHM throws an error:
    if fwhm_delay < 0.0:
        raise ValueError("fwhm_delay could not be calculated")

    # get the mean the wavelengths
    mean_shg_profile = torch.mean(symmetric_shg_matrix, dim=0)
    # get the full width half mean of the wavelengths
    fwhm_wavelength = calcFWHM(mean_shg_profile, wavelength_axis)
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
            (wavelength_axis.min(), wavelength_axis.max())
            )
    # get the subset of 'resampled_wavelength_axis', that is within the 'index_range_new_wavelength'
    resampled_wavelength_axis_subset = resampled_wavelength_axis[
            index_range_new_wavelength[0]:index_range_new_wavelength[1]
            ]
    
    # get noise level from first and last three columns of the original SHG-matrix
    noise = torch.cat((shg_matrix[:, :3].flatten(), shg_matrix[:, -3:].flatten()))
    # fill the background of the new SHG-matrix with the noise
    resampled_shg_matrix = torch.std(noise) * torch.randn(nTarget, nTarget) + torch.mean(noise)
    # resampled_shg_matrix = torch.zeros(nTarget, nTarget)

    # # Windowing function to reduce the trails at the end of the original SHG-matrix
    # # There would be a hard limit between the original SHG-matrix and the background
    # symmetric_shg_matrix = windowSHGmatrix(
    #         shg_matrix = symmetric_shg_matrix, 
    #         dimension = 0   # delay dimension
    #         )
    
    # 2D interpolate
    # initialize the interpolator
    interpolator = RectBivariateSpline(
            symmetric_delay_axis.numpy(),
            wavelength_axis.numpy(),
            symmetric_shg_matrix.numpy()
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
    resampled_shg_matrix = torch.from_numpy(resampled_shg_matrix.T)

    resampled_shg_matrix = windowSHGmatrix(
            shg_matrix = resampled_shg_matrix, 
            dimension = 0,    # delay dimension
            standard_deviation_factor=0.025
            )
    resampled_shg_matrix = windowSHGmatrix(
            shg_matrix = resampled_shg_matrix, 
            dimension = 1,    # wavelength dimension
            standard_deviation_factor=0.03
            )
    # new header
    num_delays          = int(resampled_delay_axis.size(0))
    num_wavelength      = int(resampled_wavelength_axis.size(0))
    delta_tau           = float(resampled_delay_axis[1] - resampled_delay_axis[0])
    delta_lambda        = float(resampled_wavelength_axis[1] - resampled_wavelength_axis[0])
    center_wavelength   = helper.getCenterOfAxis(resampled_wavelength_axis)
    
    new_header = torch.tensor(
            [ num_delays,
            num_wavelength,
            delta_tau,
            delta_lambda,
            center_wavelength ]
            )

    shg_data = torch.from_numpy(resampled_shg_matrix.numpy()), new_header
    resample = data.ResampleSHGmatrix(    
            config.OUTPUT_NUM_DELAYS, 
            config.OUTPUT_TIMESTEP, 
            config.OUTPUT_NUM_WAVELENGTH,
            config.OUTPUT_START_WAVELENGTH,
            config.OUTPUT_END_WAVELENGTH)

    # _, new_header, resampled_shg_matrix, resampled_delay_axis, resampled_wavelength_axis = resample(shg_data)
    # resampled_shg_matrix = helper.normalizeSHGmatrix(resampled_shg_matrix[0,:,:])
    
    # Create a figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

    # FIGURE 1
    # fix the axes to a specific exponent representation
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(-15,-15))    # use 1e-15 as exponent for x axis
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9))      # use 1e-9  as exponent for y axis
    # Plot the SHG-matrix
    c1 = ax1.pcolormesh(
        delay_axis.numpy(),
        wavelength_axis.numpy(),
        shg_matrix.numpy().T,
        shading='auto',
        # norm=LogNorm(vmin=1e-10, vmax=float( resampled_shg_matrix.max() ))
        )

    # Add labels and title
    # fig.colorbar(c1, label='Intensity')
    ax1.set_ylabel("Wavelength [m]")
    ax1.set_xlabel("Time [s]")
    ax1.set_title("Original SHG-Matrix")
    
    fig.colorbar(c1, ax=ax1, label='Intensity')
    # FIGURE 2
    # fix the axes to a specific exponent representation
    ax2.ticklabel_format(axis="x", style="sci", scilimits=(-15,-15))    # use 1e-15 as exponent for x axis
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9))      # use 1e-9  as exponent for y axis
    # Plot the SHG-matrix
    c2 = ax2.pcolormesh(
        resampled_delay_axis.numpy(),
        resampled_wavelength_axis.numpy(),
        resampled_shg_matrix.numpy().T,
        shading='auto',
        # norm=LogNorm(vmin=1e-10, vmax=float( resampled_shg_matrix.max() ))
        )

    # Add labels and title
    # fig.colorbar(c2, label='Intensity')
    ax2.set_ylabel("Wavelength [m]")
    ax2.set_xlabel("Time [s]")
    ax2.set_title("Preprocessed SHG-Matrix")

    fig.colorbar(c2, ax=ax2, label='Intensity')

    # fix the axes to a specific exponent representation
    ax3.ticklabel_format(axis="x", style="sci", scilimits=(-15,-15))    # use 1e-15 as exponent for x axis
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9))      # use 1e-9  as exponent for y axis
    # Plot the SHG-matrix
    c3 = ax3.pcolormesh(
        resampled_delay_axis.numpy(),
        resampled_wavelength_axis.numpy(),
        resampled_shg_matrix.numpy().T,
        shading='auto',
        norm=LogNorm(vmin=1e-10, vmax=float( resampled_shg_matrix.max() ))
        )

    # Add labels and title
    # fig.colorbar(c3, label='Intensity')
    ax3.set_ylabel("Wavelength [m]")
    ax3.set_xlabel("Time [s]")
    ax3.set_title("Preprocessed SHG Matrix (Logarithmic)")

    fig.colorbar(c3, ax=ax3, label='Intensity')

    # Show the plot
    plt.show()

    return resampled_shg_matrix, new_header

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

'''
preprocessFromPath()

Description:
    Preprocess the measured SHG-matrix from the path of it's file
Inputs:
    path        -> [string] path of the file the SHG-matrix and it's header are saved in
    N           -> [int] size of the SHG-matrix is (N x N) 
Outputs:
    shg_matrix  -> [tensor] preprocessed SHG-matrix
    header      -> [tensor] header of the preprocessed SHG-matrix
'''
def preprocessFromPath(path, N = 256):
    reader = data.ReadSHGmatrix()
    shg_data = reader(path)
    input_shg_matrix, input_header = shg_data
    shg_matrix, header = preprocessRawShgMatrix(input_shg_matrix, input_header, N)

    return shg_matrix, header

def preprocess(shg_path, output_path, N = 256):
    # extract the filename of the SHG-matrix
    shg_filename = os.path.basename(shg_path)
    # split the file name into the file name itself and its extension
    name, extension = os.path.splitext(shg_filename)
    # create the filename for the preproocessed SHG-matrix
    shg_preproc_filename = f"{name}_preproc{extension}"
    # create the full path of the output SHG-matrix
    full_output_path = os.path.join(output_path, shg_preproc_filename)

    shg_matrix, header = preprocessFromPath(shg_path, N)
    
    # preprocess the header
    number_delays       = int(header[0])
    number_wavelengths  = int(header[1])
    delta_tau           = round( float(header[2]) / c.femto, 4)
    delta_lambda        = round( float(header[3]) / c.nano, 4)
    center_wavelength   = round( float(header[4]) / c.nano, 4)
    # place the header information into a string
    header_line = f"{number_delays} {number_wavelengths} {delta_tau} {delta_lambda} {center_wavelength}"
    shg_lines = '\n'.join(' '.join(map(str, row.tolist() )) for row in shg_matrix)
    
    with open(full_output_path, 'w') as file:
        file.write(header_line + '\n')
        file.write(shg_lines + '\n')

    return full_output_path

'''
prepare()

Description:
    Expects dataset directory to have the following format:
        dataset_directory
        ├─ raw
        │  ├─ simulated           (these need the data in them)
        │  └─ experimental        (these need the data in them)
        └─ preproc
           ├─ simulated           (empty)
           └─ experimental        (empty)
    The raw simulated data has all it's datapoints in seperate directories,
    while the raw experimental data has all datapoints in one directory.
    All other directories should be empty
'''
def prepare(dataset_directory, experimental_blacklist_path):
    logger.info(f"dataset_directory           = {dataset_directory}")
    # create variables for needed paths
    raw_path        = os.path.join(dataset_directory, "raw")
    logger.info(f"raw path                    = {raw_path}")
    preproc_path    = os.path.join(dataset_directory, "preproc")
    logger.info(f"preproc path                = {preproc_path}")
    raw_simulated_path          = os.path.join(raw_path,"simulated")
    raw_experimental_path       = os.path.join(raw_path,"experimental")  
    logger.info(f"raw experimental path       = {raw_experimental_path}")
    logger.info(f"raw simulated path          = {raw_simulated_path}")
    preproc_simulated_path      = os.path.join(preproc_path,"simulated")  
    preproc_experimental_path   = os.path.join(preproc_path,"experimental")  
    logger.info(f"preproc experimental path   = {raw_experimental_path}")
    logger.info(f"preproc simulated path      = {raw_simulated_path}")

    # remove blacklisted spectrograms from './raw/experimental' directory
    removeBlacklistFromDirectory(
            blacklist_path= experimental_blacklist_path,
            directory_path= raw_experimental_path
            )
    # get number of experimental datapoints
    experimental_directories, experimental_files = helper.countFilesAndDirectories(directory_path=raw_experimental_path)
    experimental_elements = experimental_directories + experimental_files
    logger.info(f"Total amount of elements in '{raw_experimental_path}': {experimental_elements}")

    # get number of simulated datapoints
    simulated_directories, simulated_files = helper.countFilesAndDirectories(directory_path=raw_simulated_path)
    simulated_elements = simulated_directories + simulated_files
    logger.info(f"Total amount of elements in '{raw_simulated_path}': {simulated_elements}")
    
    # create needed subdirectories for each datapoint in './preproc/experimental'
    helper.createSubdirectories(
            base_path= preproc_experimental_path,
            name_string="s",
            number_directories= experimental_elements
            )

    # create needed subdirectories for each datapoint in './preproc/simulated'
    helper.createSubdirectories(
            base_path= preproc_simulated_path,
            name_string="s",
            number_directories= simulated_elements
            )

    # get the minimum and maximum wavelength of simulated data
    sim_min_delay, sim_max_delay, sim_min_wavelength, sim_max_wavelength = getDatasetInformation(
            data_directory=raw_simulated_path,
            matrix_filename="as_gn00.dat"
            )
    logger.info(f"Minimum Delay Simulated         = {sim_min_delay}")
    logger.info(f"Maximum Delay Simulated         = {sim_max_delay}")
    logger.info(f"Minimum Wavelength Simulated    = {sim_min_wavelength}")
    logger.info(f"Maximum Wavelength Simulated    = {sim_max_wavelength}")

    # get the minimum and maximum wavelength of experimental data
    exp_min_delay, exp_max_delay, exp_min_wavelength, exp_max_wavelength = getDatasetInformation(
            data_directory=raw_experimental_path
            )
    logger.info(f"Minimum Delay Experimental      = {exp_min_delay}")
    logger.info(f"Maximum Delay Experimental      = {exp_max_delay}")
    logger.info(f"Minimum Wavelength Experimental = {exp_min_wavelength}")
    logger.info(f"Maximum Wavelength Experimental = {exp_max_wavelength}")

    # write to info file

    # create a csv file which sorts the simulated data by TBD_{rms}

    # create subdirectory for plots of the preprocessed SHG-matrixes

    ''' MOVE BELOW TO SECOND FUNCTION'''
    # preprocess the experimental data

    # create plots of the original and preprocessed experimental SHG-matrixes

    # preprocess the simulated data

    # create plots of the original and preprocessed simulated SHG-matrixes

'''
removeBlacklistFromDirectory()

Description:
    Remove files in a blacklist from the specified directory
Inputs:
    blacklist_path  -> [string] Path to blacklist file
    directory_path  -> [string] Path to directory
Outputs:
    deleted_files   -> [array of strings] Filename of files that got deleted
    failed_files    -> [array of strings] Filename of files that weren't deleted
'''
def removeBlacklistFromDirectory(blacklist_path, directory_path):
    deleted_files = []
    failed_files = []
    logger.info(f"Deleting files listed in blacklist '{blacklist_path}' from '{directory_path}' directory!")
    try:
        # Read the blacklist
        logger.info(f"Reading blacklist '{blacklist_path}'...")
        with open(blacklist_path, newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            blacklisted_files = [row[0] for row in reader]
    except Exception as e:
        logger.info(f"Reading blacklist unsuccesful: {e}")
        
    # itterate over the blacklist and remove the files
    logger.info(f"Removing Files from '{directory_path}'...")
    for file_name in blacklisted_files:
        # get full path of selected file
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            try:
                # remove file from directory, if it exists
                os.remove(file_path)
                deleted_files.append(file_name)
            except Exception as e:
                # file doesn't exist
                logger.info(f"Failed to delete {file_name}: {e}")
                failed_files.append(file_name)
        else:
            failed_files.append(file_name)
    logger.info("Finished deletion process.")
    
    return deleted_files, failed_files

'''
preprocess_directory()
'''
def preprocess_directory(main_directory, target_directory, matrix_name=None, label_name=None):
    # get files+directories in the target directory
    target_files = os.listdir(target_directory)
    target_count = len(target_files)

    # Determine if main_dir contains files or directories
    entries= os.listdir(main_directory)
    if all(os.path.isfile(os.path.join(main_directory, entry)) for entry in entries):
        # Case 1: Files are directly inside the main directory (experimental data)
        if len(entries) != target_count:
            raise ValueError(f"Number of files in '{main_directory}' ({len(entries)}) "
                             f"does not match number of target subdirectories ({target_directory}).")
        # itterate over files and preprocess them
        for index, file_name in enumerate(entries):
            file_path = os.path.join(main_directory, file_name)
            logger.info(f"Preprocessing file {index + 1}/{target_count}: {file_path}")

    elif all(os.path.isdir(os.path.join(main_directory, entry)) for entry in entries):
        # Case 2: Files are inside subdirectories in the main directory (simulated data)
        if len(entries) != target_count:
            raise ValueError(f"Number of subdirectories in '{main_directory}' ({len(entries)}) "
                             f"does not match number of target subdirectories ({target_directory}).")

        # itterate over subdirectories and preprocess the files inside
        for index, subdirectory in enumerate(entries):
            subdirectory_path = os.path.join(main_directory, subdirectory)
            target_subdirectory_path = os.path.join(target_directory, subdirectory)
            matrix_path = os.path.join(subdirectory_path, matrix_name)
            label_path = os.path.join(subdirectory_path, label_name)

            if not os.path.isfile(matrix_path):
                raise ValueError(f"File '{matrix_name}' not found in subdirectory")
            if not os.path.isfile(label_path):
                raise ValueError(f"File '{label_name}' not found in subdirectory")

            logger.info(f"Preprocessing file {index + 1}/{target_count}: {matrix_path}")

            # copy the label
            destination_path = os.path.join(target_subdirectory_path, label_name)
            try:
                shutil.copy(label_path, destination_path)
                logger.info(f"Copied '{label_path}' to '{destination_path}'")
            except OSError as e:
                raise OSError(f"Failed to copy file to '{destination_path}': {e}")
    else:
        raise ValueError(f"The directory '{main_directory}' contains a mix of files and subdirectories or is empty")

'''
getDatasetInformation()

Description:
    Get the smallest and largest delay and wavelength values from the dataset
'''
def getDatasetInformation(data_directory, matrix_filename=None):
    '''
    Inputs:
        data_directory  -> Directory which contains the data subdirectories [string]
    Outputs:
        min_delay       -> [float] Minimum delay value in dataset
        max_delay       -> [float] Maximum delay value in dataset
        min_wavelength  -> [float] Minumum wavelength value in dataset
        max_wavelength  -> [float] Maximum wavelength value in dataset
    '''

    min_delay = float('inf')
    max_delay = float('-inf')
    min_wavelength = float('inf')
    max_wavelength = float('-inf')

    entries = os.listdir(data_directory)
    number_entries = len(entries)
    logger.info(f"Selected Directory = {data_directory}")
    logger.info(f"Number of entries in directory = {number_entries}")
    
    # check if all entries in the directory are files
    if all(os.path.isfile(os.path.join(data_directory, entry)) for entry in entries):
        logger.info(f"All Entries in {data_directory} are files!")
        for index, file_name in enumerate(entries):
            file_path = os.path.join(data_directory, file_name)
            # get the delay, highest and lowest wavelength
            delay_range, wavelength_highest, wavelength_lowest = getDelayWavelengthFromFile(file_path)
            # Update the maximum delay value
            if abs(delay_range) > max_delay:
                max_delay = abs(delay_range)
            # Update the maximum delay_value
            if abs(delay_range) < min_delay:
                min_delay = abs(delay_range)
             
            # Update maximum and minimum wavelengths
            if wavelength_highest > max_wavelength:
                max_wavelength = wavelength_highest
            if wavelength_lowest < min_wavelength:
                min_wavelength = wavelength_lowest

    # check if all entries in the directory are subdirectories
    elif all(os.path.isdir(os.path.join(data_directory, entry)) for entry in entries):
        logger.info(f"All Entries in {data_directory} are subdirectories!")
        # itterate over all subdirectories
        for index, subdirectory in enumerate(entries):
            # get the subdirectory path and also the filepath
            subdirectory_path = os.path.join(data_directory, subdirectory)
            if index == 0:
                logger.info(f"Index = {index} / {number_entries}")
                logger.info(f"subdirectory = {subdirectory}")
                logger.info(f"Matrix filename = {matrix_filename}")
            file_path = os.path.join(subdirectory_path, matrix_filename)
            # get the delay, highest and lowest wavelength
            delay_range, wavelength_highest, wavelength_lowest = getDelayWavelengthFromFile(file_path)
            # Update the maximum delay value
            if abs(delay_range) > max_delay:
                max_delay = abs(delay_range)
            # Update the minimum delay_value
            if abs(delay_range) < min_delay:
                min_delay = abs(delay_range)
                
            # Update maximum and minimum wavelengths
            if wavelength_highest > max_wavelength:
                max_wavelength = wavelength_highest
            if wavelength_lowest < min_wavelength:
                min_wavelength = wavelength_lowest
    else:
        raise ValueError(f"The directory '{data_directory}' contains a mix of files and subdirectories or is empty")
                            
    # Print the results
    logger.info(f"Min Delay: {min_delay}")
    logger.info(f"Max Delay: {max_delay}")
    logger.info(f"Min Wavelength: {min_wavelength}")
    logger.info(f"Max Wavelength: {max_wavelength}")

    return min_delay, max_delay, min_wavelength, max_wavelength

def getDelayWavelengthFromFile(path):

    if not os.path.exists(path):
        raise OSError(f"The file '{path}' does not exist!")

    with open(path, 'r') as file:
        # Read the first two lines
        lines = [file.readline(), file.readline()]
    
        # Extract the header
        header = processHeader(lines)
    
        # Extract the required values
        number_delays = header[0]  # Third element
        number_wavelength = header[1]  # Second element
        delta_tau = header[2]  # Third element
        wavelength_step = header[3]  # Fourth element
        center_wavelenght = header[4]  # Fifth element

        # Calculate wavelength related values
        wavelength_range = (number_wavelength // 2) * wavelength_step
        wavelength_highest = center_wavelenght + wavelength_range
        wavelength_lowest = center_wavelenght - wavelength_range
        
        delay_range = (number_delays // 2) * delta_tau

        return delay_range, wavelength_highest, wavelength_lowest

'''
processHeader()

Extract the header information from the first two lines of the file.
If the first line has fewer than 5 elements, the header is 2 lines.
Otherwise, it's just the first line.
'''
def processHeader(lines):
    """
    Inputs:
        lines   -> the first two lines of the spectrogram file
    Outputs:
        header  -> the header of the spectrogram file
    """
    header_line_1 = lines[0].split()
    
    if len(header_line_1) < 5:
        header_line_2 = lines[1].split()
        header = header_line_1 + header_line_2
    else:
        header = header_line_1

    if len(header) < 5:
        logger.error("ERROR: Header is has to many elements")
    
    # Convert elements to float for numeric calculations
    header = [float(x) for x in header]
    
    return header
