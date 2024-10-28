'''
helper.py Module

Module containing various helper functions
'''
#############
## Imports ##
#############
import logging
import os
import pandas as pd
import numpy as np
from pandas.core import resample
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.models as models
from torchvision import transforms

from modules import config
from modules import preprocessing
from modules import constants as c
logger = logging.getLogger(__name__)

'''
CustomDenseNetReconstruction()
Description:
    Custom DenseNet class for reconstructiong the time domain pulse from SHG-matrix
'''
class CustomDenseNetReconstruction(nn.Module):
    def __init__(self, num_outputs=512):
        '''
        Inputs:
            num_outputs -> [int] Number of outputs of the DenseNet
        '''
        super(CustomDenseNetReconstruction, self).__init__()
        # Load pretrained DenseNet
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # self.densenet = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        # Get the number of features before the last layer
        num_features = self.densenet.classifier.in_features
        # Create a Layer with the number of features before the last layer and 256 outputs (2 arrays of 128 Elements)
        self.densenet.classifier = nn.Linear(num_features, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_outputs)
        # self.densenet.classifier = nn.Linear(num_features, num_outputs)
        self.num_outputs = num_outputs

        # initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu') # useful before relu
        nn.init.zeros_(self.fc1.bias)  # Initialize biases to zero
        nn.init.xavier_normal_(self.fc2.weight) # useful before tanh
        nn.init.zeros_(self.fc2.bias)  # Initialize biases to zero

    def forward(self, shg_matrix):
        '''
        Forward pass through the DenseNet
        Input:
            shg_matrix      -> [tensor] SHG-matrix
        Output:
            x               -> [tensor] predicted output
        '''
        # get the output of the densenet
        x = self.densenet(shg_matrix)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x = torch.relu(x)
        
        # use tanh activation function to scale the output to [-1, 1] and then scale it (intensity)
        x = torch.tanh(x) 
        return x

'''
LoadDatasetReconstruction()

Description:
    Custom Dataloader Class
'''
class LoadDatasetReconstruction(Dataset):
    def __init__(self, path, label_filename, shg_filename, tbdrms_file, tbdrms_threshold, transform=None, target_transform=None):
        '''
        Inputs:
            path                -> [string] root directory containing all data subdirectories
            label_filename      -> [string] file name in which labels are stored
            shg_filename        -> [string] file name in which SHG-matrixes are stored
            tbdrms_file         -> [string] path to the CSV file containing subdirectory names and TBDrms values
            tbdrms_threshold    -> [float] maximum allowed TBDrms value
            transform           -> transform used on the SHG-Matrix
            target_transform    -> transforms used on labels
        '''
        self.path = path                            # root directory containing all data subdirectories
        self.label_filename = label_filename        # file name in which labels are stored
        self.shg_filename = shg_filename            # file name in which SHG-matrix are stored
        self.target_transform = target_transform    # transforms used on labels
        self.tbdrms_threshold = tbdrms_threshold    # max allowed TBDrms value
        self.transform = transform                  # transform used on SHG-matrixes
        self.target_transform = target_transform    # transforms used on labels

        self.data_dirs = os.listdir(self.path)      # list all subdirectories in the root directory
        
        # Load the TBDrms file
        tbdrms_data = pd.read_csv(tbdrms_file)
        
        # Initialize an empty list for storing valid directories
        valid_data_dirs = []

        # Loop through each row of the tbdrms_data DataFrame
        for _, row in tbdrms_data.iterrows():
            # Get the subdirectory name (assumed to be in the first column) and the TBDrms value (fourth column)
            subdirectory = row.iloc[0]  # First column
            tbdrms_value = row.iloc[3]  # Fourth column (TBDrms value)

            # Only add subdirectory to list if TBDrms value is within the threshold
            if tbdrms_value <= tbdrms_threshold: 
                valid_data_dirs.append(subdirectory)

        # Store the valid subdirectories
        self.data_dirs = valid_data_dirs
        
    def __len__(self):
        '''
        Description:
            Returns the number of data subdirectories (number of SHG-matrixes) [int]
        '''
        return len(self.data_dirs)              # return the number of data subdirectories

    def __getitem__(self, index):
        '''
        Description:
            Returns SHG-matrix and label of given index
        Inputs:
            index           -> [int] Index of SHG-matrix/label to be returned
        Outputs:
            output_spec     -> [tensor] SHG-matrix of given index
            label           -> [tensor] Label of given index
        '''
        data_dir = self.data_dirs[index]    # get the subdirectory for the given index
        label_path = os.path.join(self.path, data_dir, self.label_filename) # construct the full path to the label file
        shg_path = os.path.join(self.path, data_dir, self.shg_filename)     # construct the full path to the SHG-matrix file

        if self.transform:
            _, header, output_shg, _, _ = self.transform(shg_path)
        else:
            output_shg = torch.tensor(pd.read_csv(shg_path, header=None, engine='python').values, dtype=torch.half).unsqueeze(0)
            header = []

        if self.target_transform:
            label = self.target_transform(label_path)
        else:
            label = torch.tensor(pd.read_csv(label_path, header=None, engine='python').values).unsqueeze(0)
    
        # create a SHG-matrix with 3 identical channels
        # output_shg = output_shg.unsqueeze(0)  # add another dimension to the tensor
        # output_shg = output_shg.repeat(3,1,1) # repeat the SHG-matrix 3 times (3,h,w)

        # ensure correct output data type
        if not isinstance(output_shg, torch.Tensor):
            output_shg = torch.tensor(output_shg)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        return output_shg, label, header

'''
ReadShgMatrix()

Description:
    Read the SHG-matrix and it's header from a given path
'''
class ReadSHGmatrix(object):
    def __init__(self):
        pass

    def __call__(self, path):
        '''
        Inputs: 
            path        -> [string] path to the shg_matrix
        Outputs:
            shg_data    -> [shg_matrix [tensor], header [list]] Original (not resampled) SHG-Matrix and Header 
        '''
        # Constants 
        NUM_HEADER_ELEMENTS = 5

        #########################
        ## Read Header of file ##
        #########################
        with open(path, mode='r') as f:     # open the file in read mode
            first_line = f.readline().strip('\n')   # read the first line
            second_line = f.readline().strip('\n')  # read the second line
        
        first_line_len = len(first_line.split() ) # number of elements in the first line
        second_line_len = len(second_line.split() ) # number of elements in the second line
        
        # check if 1st and 2nd line have 5 Elements in total
        if(first_line_len + second_line_len != NUM_HEADER_ELEMENTS):
            # check if 1st line has 5 Elements in total
            if(first_line_len != NUM_HEADER_ELEMENTS):
                # the file has the wrong format
                logger.error(f"Number of Header Elements != {NUM_HEADER_ELEMENTS}")
                return
            else:
                # fist line has 5 Elements -> write into header
                header = first_line.split()
                num_rows_skipped = 1 # header is in first row
        else:
            # first 2 lines have 5 Elements in total -> write into header
            header = first_line.split() + second_line.split()
            num_rows_skipped = 2 # header is in first 2 rows
        
        shg_header = [
                int(header[0]),             # number of delay samples
                int(header[1]),             # number of wavelength samples
                float(header[2]) * c.femto, # time step per delay [s]
                float(header[3]) * c.nano,  # wavelength step per sample [m]
                float(header[4]) * c.nano   # center wavelength [m]
                ]
        #####################
        ## Read SHG-matrix ##
        #####################
        shg_df = pd.read_csv(path, sep='\\s+', skiprows=num_rows_skipped, header=None, engine='python')
        shg_matrix = shg_df.to_numpy()              # convert to numpy array 
        shg_matrix = torch.from_numpy(shg_matrix)   # convert to tensor

        shg_data = [shg_matrix, shg_header]
        return shg_data

'''
ResampleSHGmat()

Description:
    Transform Class that resamples SHG-matrix to use the same axis and size
'''
class ResampleSHGmatrix(object):

    def __init__(self, num_delays_out, delay_step_out, num_wavelength_out, start_wavelength_out, end_wavelength_out):
        '''
        Inputs:
            num_delays_out          -> [int] Number of delay values the resampled SHG-matrix will have
            delay_step_out          -> [float] Length of time between delays [s]
            num_wavelength_out      -> [int] Number of frequency values the resampled SHG-matrix will have
            start_wavelength_out    -> [float] Lowest wavelength value of the resampled SHG-matrix [m]
            end_wavelength_out      -> [float] Highest wavelength value of the resampled SHG-matrix [m]
        '''
        self.output_number_rows = num_delays_out
        output_delay_step = delay_step_out
        self.output_number_cols = num_wavelength_out
        output_start_wavelength = start_wavelength_out # the first element of the frequency output axis
        output_stop_wavelength = end_wavelength_out # the last element of the frequency output axis 
        output_start_delay = -(self.output_number_rows // 2) * output_delay_step # calculate time at which the output time axis starts
        output_end_delay = output_start_delay + (self.output_number_rows -1) * output_delay_step # calculate the last element of the output time axis 

        ######################## Used here to save time later
        ## Define output axis ##
        ########################
        self.output_delay = torch.linspace(output_start_delay, output_end_delay, self.output_number_rows )  # create array that corresponds to the output time axis
        self.output_wavelength = torch.linspace(output_start_wavelength, output_stop_wavelength, self.output_number_cols)    # create array that corresponds tot the output wavelength axis

        # ensure all tensors have the same type (float32)
        self.output_delay = self.output_delay.float()
        self.output_wavelength = self.output_wavelength.float()
        # initialize normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, shg_data):
        '''
        Description:
            Takes path of a SHG-matrix and resamples it to the configured size and range of time/wavelength 
        Inputs:
            shg_data                -> [shg_matrix [tensor], header [list]] Original (not resampled) SHG-Matrix and Header 
        Outputs:
            shg_resampled           -> [tensor] Resampled SHG-matrix 
            self.output_delay       -> [tensor] Resampled delay axis
            self.output_wavelength  -> [tensor] Resampled wavelength axis
        '''
        shg_original, header = shg_data
        device = shg_original.device
        # ensure all tensors are of the same type (float32)
        shg_original = shg_original.float()

        num_delays =            header[0]
        num_wavelength =        header[1]
        delay_step =            header[2]
        wavelength_step =       header[3]
        center_wavelength =     header[4]

        # create input_delay_axis and input_wavelength_axis
        input_delay_axis = preprocessing.generateAxis(N=num_delays, resolution = delay_step, center=0.0)
        input_wavelength = preprocessing.generateAxis(N=num_wavelength, resolution = wavelength_step, center = center_wavelength)

        # get minimum and maximum values of the output_delay and output_wavelength tensors
        output_wavelength_min = self.output_wavelength.min()
        output_wavelength_max = self.output_wavelength.max()

        output_delay_min = self.output_delay.min()
        output_delay_max = self.output_delay.max()

        # normalize the output delay and frequencies to [-1,1]
        normalized_output_delay      = 2 * (self.output_delay - output_delay_min) / (output_delay_max - output_delay_min) - 1 
        normalized_output_wavelength = 2 * (self.output_wavelength - output_wavelength_min) / (output_wavelength_max - output_wavelength_min) - 1  
        # create meshgrid for output delay and wavelength
        grid_delay, grid_wavelength = torch.meshgrid(normalized_output_delay,
                                              normalized_output_wavelength,
                                              indexing='ij')
        # grid sample needs the shape [H, W, 2]
        grid = torch.stack((grid_wavelength, grid_delay), dim=-1).unsqueeze(0)
        
        # reshape the SHG-matrix  to [1, 1, H, W] for grid_sample
        original_shg = original_shg.unsqueeze(0).unsqueeze(0)
        
        resampled_shg = F.grid_sample(
                original_shg.float(),
                grid.float(),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
                )

        # get shape [H,W]
        resampled_shg = resampled_shg.squeeze(0).squeeze(0)
        # get shape [channel, H,W]
        resampled_shg = resampled_shg.unsqueeze(0)  # add another dimension to the tensor
        # repeat for 3 channels
        resampled_shg = resampled_shg.repeat([3,1,1]) # repeat the SHG-matrix 3 times (3,h,w)

        # normalize
        resampled_shg = self.normalize(resampled_shg)
        
        return original_shg, header, resampled_shg, self.output_delay, self.output_wavelength

'''
ReadLabelFromEs()

Description:
    Read labels (real and imag part) from Es.dat
'''
class ReadLabelFromEs(object):
    def __init__(self, number_elements):
        '''
        Inputs:
            number_elements -> [int] Number of elements in the real and imaginary part array each
        '''
        self.number_elements = number_elements

    def __call__(self, path):    
        '''
        Read Ek.dat file and place columns in arrays
        Inputs:
            path    -> [string] path to Ek.dat
        Outputs:
            label   -> [tensor] list of arrays containing real and imag part of time signal
            [
                real_part   -> [tensor] Array containing real part of time signal
                imag_part   -> [tensor] Array containing imaginary part of time signal
            ]
        '''
        # read the dataframe
        dataframe = pd.read_csv(path,sep='  ', decimal=",", header=None, engine='python')     # sep needs to be 2 spaces
 
        # get real and imaginary part from dataframe
        _, _, _, real, imag = dataframe
        real = real.to_numpy()
        imag = imag.to_numpy()

        # Resample to fit correct number of elements
        if ( len(real) + len(imag) ) != self.number_elements:
            original_indicies = np.linspace(0, len(real) - 1, num=len(real))
            new_indicies = np.linspace(0, len(real) - 1, num=self.number_elements)
            interpolationFuncReal = interp1d(original_indicies, real, kind='linear')
            interpolationFuncImag = interp1d(original_indicies, imag, kind='linear')
            real = interpolationFuncReal(new_indicies)
            imag = interpolationFuncImag(new_indicies)
        
        # create a tensor from real and imaginary parts
        label = np.concatenate( (real, imag), axis=0)
        label = torch.from_numpy(label)
        return label

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
Scaler()

'''
class Scaler(object):
    def __init__(self, number_elements, max_real, max_imag):
        '''
        Inputs:
            number_elements -> [int] Number of elements in each the real and imaginary part of the array
            max_intensity   -> [float] >= maximum real part in dataset
            max_phase       -> [float] >= maximum imag part in dataset
        '''
        self.number_elements = number_elements
        self.max_real = max_real
        self.max_imag = max_imag

    def scale(self, label):
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            label           -> [tensor] List of arrays containing real and imaginary part of time signal
        Outputs:
            scaled_label    -> [tensor] List of arrays containing real and imaginary part of the time signal scaled to [-1,1]
        '''
        # get real and imaginary part
        real_part = label[:self.number_elements]
        imag_part = label[self.number_elements:]
        # scale the real and imaginary part
        scaled_real = real_part / self.max_real
        scaled_imag = imag_part / self.max_imag
        # concat them back into label
        scaled_label = torch.cat([scaled_real, scaled_imag])

        return scaled_label
        
    def unscale(self, scaled_label):
        '''
        Unscales the values of the phase to [-1,1]
        Inputs:
            scaled_label    -> [tensor] List of arrays containing the real and imaginary part scaled to [-1,1]
        Outputs:
            label           -> [tensor] List of arrays containing the real and imaginary part of time signal
        '''
        # get real and imaginary part
        scaled_real_part = scaled_label[:self.number_elements]
        scaled_imag_part = scaled_label[self.number_elements:]

        # scale the real and imaginary part
        unscaled_real_part = scaled_real_part * self.max_real
        unscaled_imag_part = scaled_imag_part * self.max_imag

        # concat them back into label
        unscaled_label = torch.cat([unscaled_real_part, unscaled_imag_part])

        return unscaled_label

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
frequency_axis_from_header()

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
    wavelength_axis = preprocessing.generateAxis(
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
getCenterOfAxis

Description:
    return the center element of an axis
Inputs:
    axis            -> [tensor] wavelength/delay/frequency axis
Outputs:
    center_element  -> [float] center element of the axis
'''
def getCenterOfAxis(axis):
    length_axis = axis.size(0)
    # if the axis has an even number of elements
    if length_axis % 2 == 0: 
        center_index = length_axis // 2 + 1
    # if the axis has an odd number of elements
    else:
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
