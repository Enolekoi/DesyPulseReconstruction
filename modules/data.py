""" Data Module

Module containing functions and classes for loading or transforming data
"""
import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.interpolate import interp1d

from modules import constants as c
from modules import helper

logger = logging.getLogger(__name__)

'''
LoadDatasetReconstruction()

Description:
    Custom Dataloader Class
'''
class LoadDatasetReconstruction(Dataset):
    def __init__(self, path, label_filename, shg_filename, tbdrms_file, tbdrms_threshold, transform=None, target_transform=None, use_label=True,):
        '''
        Inputs:
            path                -> [string] root directory containing all data subdirectories
            label_filename      -> [string] file name in which labels are stored
            shg_filename        -> [string] file name in which SHG-matrixes are stored
            tbdrms_file         -> [string] path to the CSV file containing subdirectory names and TBDrms values
            tbdrms_threshold    -> [float] maximum allowed TBDrms value
            transform           -> transform used on the SHG-Matrix
            target_transform    -> transforms used on labels
            use_label           -> [bool] load labels from dataset
        '''
        self.path = path                            # root directory containing all data subdirectories
        self.use_label = use_label
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
        shg_path = os.path.join(self.path, data_dir, self.shg_filename)     # construct the full path to the SHG-matrix file

        if self.transform:
            _, header, output_shg, _, _ = self.transform(shg_path)
        else:
            output_shg = torch.tensor(pd.read_csv(shg_path, header=None, engine='python').values, dtype=torch.half).unsqueeze(0)
            header = []

        # ensure correct output data type
        if not isinstance(output_shg, torch.Tensor):
            output_shg = torch.tensor(output_shg)

        if self.use_label == True:
            label_path = os.path.join(self.path, data_dir, self.label_filename) # construct the full path to the label file
            if self.target_transform:
                label = self.target_transform(label_path)
            else:
                label = torch.tensor(pd.read_csv(label_path, header=None, engine='python').values).unsqueeze(0)

            # ensure correct output data type
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label)
        else:
            return output_shg, header

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
            shg_data    -> [shg_matrix [tensor], header [tensor]] Original (not resampled) SHG-Matrix and Header 
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
        shg_header = torch.tensor(shg_header)
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
        # initialize normalization with values for the densenet (https://pytorch.org/hub/pytorch_vision_densenet/)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, shg_data):
        '''
        Description:
            Takes path of a SHG-matrix and resamples it to the configured size and range of time/wavelength 
        Inputs:
            shg_data                -> [shg_matrix [tensor], header [tensor]] Original (not resampled) SHG-Matrix and Header 
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
        input_delay_axis = helper.generateAxis(
                N=num_delays,
                resolution=delay_step,
                center=0.0
                )
        input_wavelength_axis = helper.generateAxis(
                N=num_wavelength,
                resolution=wavelength_step,
                center=center_wavelength
                )

        # resample along the delay axis
        delay_resampled = torch.stack([
            helper.piecewiseLinearInterpolation(input_delay_axis,
                                         shg_original[:, i],
                                         self.output_delay)
            for i in range(shg_original.shape[1])
            ], dim=1)

        # resample along the wavelength axis
        shg_resampled = torch.stack([
            helper.piecewiseLinearInterpolation(input_wavelength_axis,
                                         delay_resampled[i, :],
                                         self.output_wavelength)
            for i in range(delay_resampled.shape[0])
            ], dim=0)

        # add another dimension to the tensor
        shg_resampled = shg_resampled.unsqueeze(0)
        # repeat the SHG-matrix for 3 times to get 3 channels (shape =[3,h,w])
        shg_resampled = shg_resampled.repeat([3,1,1])

        # normalize using weights specified for the densenet
        shg_resampled = self.normalize(shg_resampled)
        
        return shg_original, header, shg_resampled, self.output_delay, self.output_wavelength

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
        # time, intensity, phase, real, imag = dataframe.to_numpy()
        real = dataframe[3].to_numpy()
        imag = dataframe[4].to_numpy()

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
Scaler()

Scaler used for scaling the label
'''
class Scaler(object):
    def __init__(self, number_elements, max_value):
        '''
        Inputs:
            number_elements -> [int] Number of elements in each the real and imaginary part of the array
            max_value       -> [float] >= maximum label value in dataset
        '''
        self.number_elements = number_elements
        self.max_value = max_value

    def scale(self, label):
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            label           -> [tensor] Unscaled Label
        Outputs:
            scaled_label    -> [tensor] Label scaled to [-1,1]
        '''
        # scale the label
        scaled_label = label / self.max_value

        return scaled_label
        
    def unscale(self, scaled_label):
        '''
        Unscales the values of the phase to [-1,1]
        Inputs:
            scaled_label    -> [tensor] Label scaled to [-1,1]
        Outputs:
            label           -> [tensor] Unscaled Label
        '''
        # unscale the label
        unscaled_label = scaled_label * self.max_value

        return unscaled_label

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
        complex_signal = helper.removeTranslationAmbiguity(complex_signal, self.center_index)
        # remove mirrored complex conjugate ambiguity Eamb(t) = E*(-t)
        complex_signal = helper.removeConjugationAmbiguity(complex_signal, self.center_index)
        # remove absolute phase shift ambiguity Eamb(t) = E(t)*exp(j\phi0)
        complex_signal = helper.removePhaseShiftAmbiguity(complex_signal, self.center_index)

        real = complex_signal.real
        imag = complex_signal.imag

        output_label = torch.cat([real, imag])
        return output_label


