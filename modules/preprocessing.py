'''
Preprocessing
'''
#############
## Imports ##
#############
import os
import re
import shutil
import warnings
import logging
import torch
import numpy as np
import csv
from scipy.interpolate import RectBivariateSpline
from scipy.signal import windows

from modules import constants as c
from modules import config
from modules import data
from modules import helper
from modules import visualize as vis

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

    if isinstance(shg_matrix, np.ndarray):
        # convert to tensor if spectrogram is a numpy array
        shg_matrix = torch.from_numpy(shg_matrix)

    # determine the window width (equals the number of samples across the selected dimension)
    window_width = int(shg_matrix.size(dimension))
    # calculate the standard deviation used for the gaussian window
    window_standard_deviation = window_width * standard_deviation_factor
    # logger.info(f"Window width                        = {window_width}")
    # logger.info(f"Standard deviation of the window    = {window_standard_deviation}")
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
    logger.info("normalizing SHG-Matrix")
    shg_matrix = helper.normalizeSHGmatrix(shg_matrix)
    shg_matrix_original = shg_matrix
    
    # get the delay and wavelength axis
    logger.info("Getting Axis")
    delay_axis, wavelength_axis = helper.generateAxes(header)

    # if nTarget is not even
    if nTarget % 2 !=0:
        raise ValueError("nTarget must be even")

    # if the shape of the spectrogram_matrix is not even
    if shg_matrix.shape[0] != num_delays or shg_matrix.shape[1] != num_wavelength:
        warnings.warn("shg matrix and header information don't match! Exiting Funtion.", UserWarning)
        resampled_shg_matrix = shg_matrix
        new_header = header
        raise ValueError("spectrogram dimension aren't even")

    logger.info("Making SHG-matrix symmetrical")
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
    
    logger.info("Making SHG-matrix the same side in each direction of the Center of Mass")
    # devide the com_delay_index by the sum of total SHG-matrix values and round it 
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

    logger.info("Resampling 1")
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

    logger.info("Resampling 2")
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
    # noise = torch.cat((shg_matrix[:3, :].flatten(), shg_matrix[-3:, :].flatten()))
    # fill the background of the new SHG-matrix with the noise
    # resampled_shg_matrix = torch.std(noise) * torch.randn(nTarget, nTarget) + torch.mean(noise)
    # resampled_shg_matrix = torch.zeros(nTarget, nTarget)

    logger.info("Resampling 3")
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

    logger.info("Window")
    # use a windowing function
    shg_interpolated = windowSHGmatrix(
            shg_matrix = shg_interpolated, 
            dimension = 0,    # delay dimension
            standard_deviation_factor=0.08
            )
    shg_interpolated = windowSHGmatrix(
            shg_matrix = shg_interpolated, 
            dimension = 1,    # wavelength dimension
            standard_deviation_factor=0.08
            )
    # get the upper 5% of the edge
    width = shg_interpolated.size(0)
    top_percent_width = int(width * 0.1)
    left_edge = shg_interpolated[:top_percent_width, 0]
    right_edge = shg_interpolated[:top_percent_width, -1]
    noise = torch.cat([left_edge, right_edge], dim=0)
    # get noise level from first and last three columns of the original SHG-matrix
    # noise = torch.cat((shg_interpolated[:3, :].flatten(), shg_interpolated[-3:, :].flatten()))
    # noise = torch.cat((shg_interpolated[:top_5_percent_width, :].flatten(), shg_interpolated[-top_5_percent_width:, :].flatten()))
    # fill the background of the new SHG-matrix with the noise
    resampled_shg_matrix = torch.std(noise) * torch.abs(torch.randn(nTarget, nTarget)) + torch.mean(noise)
    # resampled_shg_matrix = torch.zeros(nTarget, nTarget)

    logger.info("Place into larger Matrix")
    # Embed the interpolated matrix into the larger matrix
    resampled_shg_matrix = resampled_shg_matrix.numpy()
    resampled_shg_matrix[index_wavelength_range_min:index_wavelength_range_max,\
            index_delay_range_min:index_delay_range_max] = shg_interpolated
    resampled_shg_matrix = torch.from_numpy(resampled_shg_matrix.T)
    
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

    logger.info("Final resample")
    resample_outputs = resample(shg_data)
    _, new_header, resampled_shg_matrix, resampled_delay_axis, resampled_wavelength_axis = resample_outputs
    
    # create output header
    output_num_delays = resampled_delay_axis.size(0)
    output_num_wavelength = resampled_wavelength_axis.size(0)
    output_delta_tau = resampled_delay_axis[1] - resampled_delay_axis[0]
    output_delta_lambda = resampled_wavelength_axis[1] - resampled_wavelength_axis[0]
    output_center_wavelength = helper.getCenterOfAxis(resampled_wavelength_axis)
    
    output_header = [
            output_num_delays,
            output_num_wavelength,
            output_delta_tau,
            output_delta_lambda,
            output_center_wavelength
            ]

    return resampled_shg_matrix, output_header

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
preprocess()

Description:
    Preprocess the measured SHG-matrix from the path of it's file
Inputs:
    shg_path    -> [string] path of the file where the raw SHG-matrix and it's header are saved in
    output_path -> [string] path of the file where the preprocessed SHG-matrix and it's header are saved in
    N           -> [int] size of the SHG-matrix is (N x N) 
Outputs:
    shg_matrix  -> [tensor] preprocessed SHG-matrix
    header      -> [tensor] header of the preprocessed SHG-matrix
'''

def preprocess(shg_path, output_path, N = 256):
    # initialize SHG-Matrix reader
    reader = data.ReadSHGmatrix()
    # read in the matrix and header
    shg_data = reader(shg_path)
    input_shg_matrix, input_header = shg_data
    # preprocess the raw SHG-Matrix
    shg_matrix, header = preprocessRawShgMatrix(input_shg_matrix, input_header, N)

    # print(shg_matrix.shape)
    # preprocess the header
    number_delays       = int(header[0])
    number_wavelengths  = int(header[1])
    delta_tau           = round( float(header[2]) / c.femto, 4)
    delta_lambda        = round( float(header[3]) / c.nano, 4)
    center_wavelength   = round( float(header[4]) / c.nano, 4)
    # place the header information into a string
    header_line = f"{number_delays} {number_wavelengths} {delta_tau} {delta_lambda} {center_wavelength}"
    shg_lines = '\n'.join(' '.join(map(str, row.tolist() )) for row in shg_matrix)
    
    with open(output_path, 'w') as file:
        file.write(header_line + '\n')
        file.write(shg_lines + '\n')

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
    logger.info(f"Deleted files: {deleted_files}")
    
    return deleted_files, failed_files

'''
preprocess_experimental()

Description:
    Preprocess the experimental data. It's expected, that raw_dir has the spectrogram files directly inside,
    while preproc_dir has the same amount of subdirectories
Inputs:
    raw_dir             -> [string] Path to directory containing the raw experimental SHG-matrixes
    preproc_dir         -> [string] Path to directory containing subdirectories for preprocessed SHG-matrixes
    plot_dir            -> [string] Path to directory, where the comparison of preprocessed and raw spectrograms get saved
    preproc_filename    -> [string] File name of the preprocessed SHG-matrix
    grid_size           -> [int] (optional, default=512) Grid to which the spectrograms get resampled to
'''
def preprocess_experimental(raw_dir, preproc_dir, plot_dir, preproc_filename, grid_size=512):
    # Regular expression for extracting the indices
    raw_file_pattern = re.compile(r'\D*(\d+)\.txt$')
    preproc_dir_pattern = re.compile(r's(\d+)$')
    
    # List all input files and directories
    raw_files = [f for f in os.listdir(raw_dir) if raw_file_pattern.search(f)]
    preproc_dirs = [d for d in os.listdir(preproc_dir) if preproc_dir_pattern.search(d)]

    # Extract indices and sort the files and directories by index
    raw_files_with_indices = sorted(
            [(int(raw_file_pattern.search(f).group(1)), f) for f in raw_files]
            )
    preproc_dirs_with_indices = sorted(
            [(int(preproc_dir_pattern.search(d).group(1)), d) for d in preproc_dirs]
            )
    
    # check if the number of files matches the number of directories
    if len(raw_files_with_indices) != len(preproc_dirs_with_indices):
        raise ValueError(f"Mismatch between number of input files ({len(raw_files_with_indices)}) and output directories ({len(preproc_dirs_with_indices)})!")


    # itteratae over sorted files and directories
    for(raw_index, raw_file), (preproc_index, preproc_subdir) in zip(raw_files_with_indices, preproc_dirs_with_indices):
        logger.info(f"Preprocessing Experimental Matrix {preproc_index}/{len(preproc_dirs_with_indices)}")
        logger.info(f"Raw path          = {raw_file}")
        logger.info(f"Preprocessed path = {preproc_subdir}")

        try:
            # Construct raw file path
            raw_file_path = os.path.join(raw_dir, raw_file)
            
            # Construct preproc file path
            preproc_file_path = os.path.join(preproc_dir, preproc_subdir, preproc_filename)
            
            # Ensure the preproc directory exists
            os.makedirs(os.path.dirname(preproc_file_path), exist_ok=True)
            
            # Call the preprocessing funtion
            preprocess(
                    shg_path = raw_file_path,
                    output_path = preproc_file_path,
                    N = grid_size
                    )
            
            # Plot the comparison of the raw and preprocessed SHG-matrix
            plot_filename = f"comparison_{raw_index}.png"
            plot_path = os.path.join(plot_dir, plot_filename) 
            vis.comparePreproccesSHGMatrix(
                raw_filepath=raw_file_path,
                preproc_filepath= preproc_file_path,
                save_path=plot_path
                )
        except Exception as e:
            logger.error(f"An error occurred: {e}")

'''
preprocess_simulated()

Description:
    Preprocess the simulated data. It's expected, that raw_dir has the spectrogram files directly inside,
    while preproc_dir has the same amount of subdirectories
Inputs:
    raw_dir                     -> [string] Path to directory containing subdirectories with the raw simulated data
    preproc_dir                 -> [string] Path to directory containing subdirectories for preprocessed data
    plot_dir                    -> [string] Path to directory, where the comparison of preprocessed and raw spectrograms get saved
    raw_filename_matrix         -> [string] Path to directory containing subdirectories with the raw simulated data
    raw_filename_label          -> [string] Path to directory containing subdirectories with the raw simulated data
    preproc_filename_matrix     -> [string] File name of the preprocessed SHG-matrix
    preproc_filename_label      -> [string] File name of the preprocessed SHG-matrix
    grid_size                   -> [int] (optional, default=512) Grid to which the spectrograms get resampled to
'''
def preprocess_simulated(
        raw_dir, preproc_dir, plot_dir,
        raw_filename_matrix, raw_filename_label,
        preproc_filename_matrix, preproc_filename_label,
        grid_size=512):

    # Regular expression for extracting the indices
    raw_dir_pattern = re.compile(r's(\d+)$')
    preproc_dir_pattern = re.compile(r's(\d+)$')
    
    # List all input files and directories
    raw_dirs = [d for d in os.listdir(raw_dir) if raw_dir_pattern.search(d)]
    preproc_dirs = [d for d in os.listdir(preproc_dir) if preproc_dir_pattern.search(d)]

    # Extract indices and sort the files and directories by index
    raw_dirs_with_indices = sorted(
            [(int(raw_dir_pattern.search(d).group(1)), d) for d in raw_dirs]
            )
    preproc_dirs_with_indices = sorted(
            [(int(preproc_dir_pattern.search(d).group(1)), d) for d in preproc_dirs]
            )
    
    # check if the number of files matches the number of directories
    if len(raw_dirs_with_indices) != len(preproc_dirs_with_indices):
        raise ValueError(f"Mismatch between number of input files ({len(raw_dirs_with_indices)}) and output directories ({len(preproc_dirs_with_indices)})!")

    # itteratae over sorted files and directories
    for(raw_index, raw_subdir), (preproc_index, preproc_subdir) in zip(raw_dirs_with_indices, preproc_dirs_with_indices):
        logger.info(f"Preprocessing Simulated Matrix {preproc_index}/{len(preproc_dirs_with_indices)}")
        logger.info(f"Raw path          = {raw_subdir}")
        logger.info(f"Preprocessed path = {preproc_subdir}")
        try:
            # Construct raw file paths
            raw_file_path_matrix = os.path.join(raw_dir, raw_subdir, raw_filename_matrix)
            raw_file_path_label = os.path.join(raw_dir, raw_subdir, raw_filename_label)
            
            # Construct preproc file paths
            preproc_file_path_matrix = os.path.join(preproc_dir, preproc_subdir, preproc_filename_matrix)
            preproc_file_path_label = os.path.join(preproc_dir, preproc_subdir, preproc_filename_label)
            
            # Ensure the preproc directory exists
            os.makedirs(os.path.dirname(preproc_file_path_matrix), exist_ok=True)
            
            logger.info(f"Copying Label")
            # copy label
            shutil.copy(raw_file_path_label, preproc_file_path_label)
            
            logger.info(f"Starting Preprocessing")
            # Call the preprocessing funtion
            preprocess(
                    shg_path = raw_file_path_matrix,
                    output_path = preproc_file_path_matrix,
                    N = grid_size
                    )
            
            # Plot the comparison of the raw and preprocessed SHG-matrix
            plot_filename = f"comparison_{raw_index}.png"
            plot_path = os.path.join(plot_dir, plot_filename)
            vis.comparePreproccesSHGMatrix(
                    raw_filepath=raw_file_path_matrix,
                    preproc_filepath= preproc_file_path_matrix,
                    save_path=plot_path
                    )
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            

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
            if index % 100 == 0:
                logger.info(f"Index = {index} / {number_entries}; subdirectory = {subdirectory}")
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

'''
writeDatasetInformationToFile()
'''
def writeDatasetInformationToFile(
        file_path,
        dataset_path,
        num_datapoints,
        min_delay, 
        max_delay, 
        min_wavelength, 
        max_wavelength, 
        grid_size=512
        ):
    half_grid = grid_size // 2
    # check if path is correct
    if os.path.isdir(os.path.dirname(file_path)):
        with open(file_path, 'w') as file:
            file.write(f"Dataset Information for '{dataset_path}'.\n")
            file.write(f"Amount of datapoints in dataset                  = {num_datapoints}\n")
            file.write(f"Smallest delay range in dataset                  = {min_delay}\n")
            file.write(f"Largest delay range in dataset                   = {max_delay}\n")
            file.write(f"Smallest delta tau in dataset ({grid_size} Grid) = {min_delay/half_grid}\n")
            file.write(f"Largest delta tau in dataset ({grid_size} Grid)  = {max_delay/half_grid}\n")
            file.write(f"Smallest wavelength in dataset                   = {min_wavelength}\n")
            file.write(f"Largest wavelength in dataset                    = {max_wavelength}\n")
        logger.info(f"File written to {file_path}!")
    else:
        raise OSError(f"Path {file_path} is incorrect!")

'''
getTBDrmsValues

Description:
    Calculate TBDrms values for all subdirectories, sort them and write them to a file

Inputs:
    data_directory  -> [string] Base directorie of the training dataset
    root_directory  -> [string] Root directorie of the project
    output_filename -> [string] Filename to write the sorted TBDrms values to
'''                            
def getTBDrmsValues(data_directory, root_directory, output_filename):
    data = []
    
    logger.info('Stepping through all subdirectories')
    # step through all directories
    for subdirectory in os.listdir(data_directory):
        # get the subdirectory path 
        subdirectory_path = os.path.join(data_directory, subdirectory)

        # Ensure it's a subdirectory and starts with 's'
        if os.path.isdir(subdirectory_path) and subdirectory.startswith('s'):
            # get the path of SimulatedPulseData.txt
            file_path = os.path.join(subdirectory_path, 'SimulatedPulseData.txt')

            # Check if 'SimulatedPulseData.txt' exists
            if os.path.exists(file_path):
                # initialize rmsT and rmsW 
                rmsT, rmsW = None, None

                # open and read 'SimulatedPulseData.txt'
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        # Extract rmsT value
                        if line.startswith('rmsT;'):
                            rmsT = float(line.split(';')[1].strip())
                        # Extract rmsW value
                        if line.startswith('rmsW;'):
                            rmsW = float(line.split(';')[1].strip())

                # Check if rmsT and rmsW have values
                if (rmsT is not None) and (rmsW is not None): 
                    TBDrms = rmsT*rmsW
                    # Store values as a tuple (subdirectory, rmsT, rmsW, TBDrms)
                    data.append([subdirectory, rmsT, rmsW, TBDrms])

    logger.info('Got all data from subdirectories')
    #####################
    ## WRITE CSV FILES ##
    #####################
    logger.info('Creating sorted CSV-file')
    sorted_data = sorted(data, key=lambda x: x[3], reverse=False)

    sorted_csv_file = os.path.join(root_directory, output_filename)
    with open(sorted_csv_file, 'w', newline='') as sorted_csvfile:
        writer = csv.writer(sorted_csvfile)
        # write header
        writer.writerow(['Directory', 'rmsT', 'rmsW', 'TBDrms'])
        # write data 
        writer.writerows(sorted_data)
'''
createPlotSubdirectories()

Description:
    Creates directories for plots of the spectrograms
Inputs:
    path    -> [string] Path of the parent directory
'''
def createPlotSubdirectories(path):
    try:
        # Create the main directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
            logger.info(f"Created plot directory at '{path}'!")
        else:
            logger.info(f"Plot directory already exists at '{path}'")
        
        # Create the subdirectories
        subdir1_path = os.path.join(path, "experimental")
        subdir2_path = os.path.join(path, "simulated")

        os.makedirs(subdir1_path, exist_ok=True)
        os.makedirs(subdir2_path, exist_ok=True)

        logger.info(f"Subdirectories 'experimental' and 'raw' created in '{path}'.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

'''
prepareDirectoriesForPreprocessing()

Description:
    Prepares the datasets and their directories for preprocessing!
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
    All other directories should be empty!

Inputs:
    dataset_directory               -> [string] path of the parent directory
    grid_size                       -> [int]
    experimental_blacklist_path     -> [string] path to the blacklist used on experimental data

'''
def prepareDirectoriesForPreprocessing(dataset_directory, grid_size, experimental_blacklist_path = None):
    # create variables for needed paths
    raw_path        = os.path.join(dataset_directory, "raw")
    preproc_path    = os.path.join(dataset_directory, "preproc")
    plots_path      = os.path.join(dataset_directory, "plots")
    raw_simulated_path          = os.path.join(raw_path,"simulated")
    raw_experimental_path       = os.path.join(raw_path,"experimental")  
    preproc_simulated_path      = os.path.join(preproc_path,"simulated")  
    preproc_experimental_path   = os.path.join(preproc_path,"experimental")  
    logger.info(f"dataset directory           = {dataset_directory}") 
    logger.info(f"plots path                  = {plots_path}") 
    logger.info(f"raw path                    = {raw_path}")
    logger.info(f"preproc path                = {preproc_path}")
    logger.info(f"raw experimental path       = {raw_experimental_path}")
    logger.info(f"raw simulated path          = {raw_simulated_path}")
    logger.info(f"preproc experimental path   = {raw_experimental_path}")
    logger.info(f"preproc simulated path      = {raw_simulated_path}")

    # remove blacklisted spectrograms from './raw/experimental' directory
    if experimental_blacklist_path is not None:
        logger.info(f"Removing files listed in '{experimental_blacklist_path}' from '{raw_experimental_path}'!")
        removeBlacklistFromDirectory(
                blacklist_path= experimental_blacklist_path,
                directory_path= raw_experimental_path
                )
    else:
        logger.info(f"No blacklist for experimental data specified!")

    # get number of experimental datapoints
    experimental_directories, experimental_files = helper.countFilesAndDirectories(directory_path=raw_experimental_path)
    experimental_elements = experimental_directories + experimental_files
    logger.info(f"Total amount of elements in '{raw_experimental_path}': {experimental_elements}")

    # get number of simulated datapoints
    simulated_directories, simulated_files = helper.countFilesAndDirectories(directory_path=raw_simulated_path)
    simulated_elements = simulated_directories + simulated_files
    logger.info(f"Total amount of elements in '{raw_simulated_path}': {simulated_elements}")
    
    # remove ./preproc/experimental/*
    helper.removeAllFromDirectory(preproc_experimental_path)
    # remove ./preproc/simulated/*
    helper.removeAllFromDirectory(preproc_simulated_path)
    # remove ./plots
    helper.removeAllFromDirectory(plots_path)

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
    logger.info(f"Minimum Delay Simulated           = {sim_min_delay} fs")
    logger.info(f"Maximum Delay Simulated           = {sim_max_delay} fs")
    logger.info(f"Minimum Wavelength Simulated      = {sim_min_wavelength} nm")
    logger.info(f"Maximum Wavelength Simulated      = {sim_max_wavelength} nm")
                                                    
    logger.info(f"Minimum delta_tau Simulated       = {sim_min_delay/(grid_size//2)} fs ({grid_size} Grid)")
    logger.info(f"Maximum delta_tau Simulated       = {sim_max_delay/(grid_size//2)} fs ({grid_size} Grid)")
    logger.info(f"Minimum delta_lambda Simulated    = {sim_min_wavelength/grid_size} nm ({grid_size} Grid)")
    logger.info(f"Maximum delta_lambda Simulated    = {sim_max_wavelength/grid_size} nm ({grid_size} Grid)")

    # get the minimum and maximum wavelength of experimental data
    exp_min_delay, exp_max_delay, exp_min_wavelength, exp_max_wavelength = getDatasetInformation(
            data_directory=raw_experimental_path
            )
    logger.info(f"Minimum Delay Experimental        = {exp_min_delay} fs")
    logger.info(f"Maximum Delay Experimental        = {exp_max_delay} fs")
    logger.info(f"Minimum Wavelength Experimental   = {exp_min_wavelength} nm")
    logger.info(f"Maximum Wavelength Experimental   = {exp_max_wavelength} nm")

    logger.info(f"Minimum delta_tau Experimental    = {exp_min_delay/grid_size} fs ({grid_size} Grid)")
    logger.info(f"Maximum delta_tau Experimental    = {exp_max_delay/grid_size} fs ({grid_size} Grid)")
    logger.info(f"Minimum delta_lambda Experimental = {exp_min_wavelength/grid_size} nm ({grid_size} Grid)")
    logger.info(f"Maximum delta_lambda Experimental = {exp_max_wavelength/grid_size} nm ({grid_size} Grid)")

    # write to info file for raw experimental data
    raw_experimental_info_path = os.path.join(raw_path, "experimental_info.txt")
    writeDatasetInformationToFile(        
        file_path = raw_experimental_info_path,
        dataset_path = raw_experimental_path,
        num_datapoints = experimental_elements,
        min_delay = exp_min_delay,  
        max_delay = exp_max_delay, 
        min_wavelength = exp_min_wavelength, 
        max_wavelength = exp_max_wavelength, 
        grid_size = 512)

    # write to info file for raw simulated data
    raw_simulated_info_path = os.path.join(raw_path, "simulated_info.txt")
    writeDatasetInformationToFile(        
        file_path = raw_simulated_info_path,
        dataset_path = raw_simulated_path,
        num_datapoints = simulated_elements,
        min_delay = sim_min_delay,  
        max_delay = sim_max_delay, 
        min_wavelength = sim_min_wavelength, 
        max_wavelength = sim_max_wavelength, 
        grid_size = 512)

    # create a csv file which sorts the simulated data by TBD_{rms}
    getTBDrmsValues(
            data_directory = raw_simulated_path,
            root_directory = dataset_directory, 
            output_filename = "TBD_rms.csv"
            )

    # create subdirectory for plots of the preprocessed SHG-matrixes
    createPlotSubdirectories(plots_path)

'''
pre
'''
def pre(dataset_directory, grid_size=512):

    preproc_path = os.path.join(dataset_directory, "preproc")
    raw_path = os.path.join(dataset_directory, "raw")

    preproc_experimental_path = os.path.join(preproc_path, "experimental")
    preproc_simulated_path = os.path.join(preproc_path, "simulated")
    raw_experimental_path = os.path.join(raw_path, "experimental")
    raw_simulated_path = os.path.join(raw_path, "simulated")

    plots_path = os.path.join(dataset_directory, "plots")
    plots_experimental_path = os.path.join(plots_path, "experimental")
    plots_simulated_path = os.path.join(plots_path, "simulated")
    # preprocess the simulated data
        # increment through datapoints and write result in /.preproc/simulated 
    preprocess_simulated(
            raw_dir=raw_simulated_path, 
            preproc_dir=preproc_simulated_path, 
            plot_dir=plots_simulated_path, 
            raw_filename_matrix = "as_gn00.dat", 
            raw_filename_label = "Es.dat", 
            preproc_filename_matrix = "as_gn00.dat", 
            preproc_filename_label = "as_gn00.dat", 
            grid_size=grid_size
            )
    # preprocess the experimental data
        # increment through datapoints and write result in /.preproc/experimental 
    preprocess_experimental(
            raw_dir=raw_experimental_path, 
            preproc_dir=preproc_experimental_path, 
            plot_dir=plots_experimental_path, 
            preproc_filename = "as_gn00.dat", 
            grid_size=grid_size
            )
