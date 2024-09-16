'''
HELPER FUNCTIONS
'''
#############
## Imports ##
#############
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os   # Dataloader, 
import torch    # Dataloader,
from torch.utils.data import Dataset    # Dataloader,
import torchvision.models as models     # Custom DenseNet
import torch.nn as nn   # Custom DenseNet
import logging

import config

logger = logging.getLogger(__name__)

# TODO:
# Implement own MSE loss function that considers ambiguities in phase retrieval (Trebino, p. 63)
'''
CustomDenseNet()
Custom DenseNet class
'''
class CustomDenseNet(nn.Module):
    def __init__(self, num_outputs=512):
        '''
        Inputs:
            num_outputs     -> Number of outputs of the DenseNet [int]
        '''
        super(CustomDenseNet, self).__init__()
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

    def forward(self, spectrogram):
        '''
        Forward pass through the DenseNet
        Input:
            spectrogram     -> input spectrogram [tensor]
        Output:
            x   -> predicted output [tensor]
        '''
        # half_size = int(self.num_outputs //2)
        # get the output of the densenet
        x = self.densenet(spectrogram)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x = torch.relu(x)
        
        # use tanh activation function to scale the output to [-1, 1] and then scale it (intensity)
        x = torch.tanh(x)
        return x

'''
TimeDomain()
Class containing information of the time domain signal
'''
class TimeDomain:
    def __init__(self, time_axis, intensity, phase, real, imag):
        '''
        Inputs:
            time_axis   -> Time axis
            intensity   -> Intensity (squared amplitude) of time domain signal
            phase       -> Phase of time domain signal
            real        -> Real part of time domain signal
            imag        -> Imaginary part of time domain signal
        '''
        self.time_axis = time_axis
        self.intensity = intensity
        self.phase = phase
        self.real = real
        self.imag = imag

'''
SimulatedDataset()
Custom Dataloader Class
'''
class SimulatedDataset(Dataset):
    def __init__(self, path, label_filename, spec_filename, tbdrms_file, tbdrms_threshold, transform=None, target_transform=None):
        '''
        Inputs:
            path                -> root directory containing all data subdirectories [string]
            label_filename      -> file name in which labels are stored [string]
            spec_filename       -> file name in which spectrograms are stored [string] 
            tbdrms_file         -> path to the CSV file containing subdirectory names and TBDrms values [string]
            tbdrms_threshold    -> maximum allowed TBDrms value [float]
            transform           -> transform used on spectrograms 
            target_transform    -> transforms used on labels
        '''
        self.path = path    # root directory containing all data subdirectories
        self.label_filename = label_filename      # file name in which labels are stored
        self.spec_filename = spec_filename        # file name in which spectrograms are stored
        self.target_transform = target_transform    # transforms used on labels
        self.tbdrms_threshold = tbdrms_threshold   # max allowed TBDrms value
        self.transform = transform              # transform used on spectrograms
        self.target_transform = target_transform    # transforms used on labels

        self.data_dirs = os.listdir(self.path)  # list all subdirectories in the root directory
        
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
        Returns the number of data subdirectories (number of spectrograms) [int]
        '''
        return len(self.data_dirs)              # return the number of data subdirectories

    def __getitem__(self, index):
        '''
        Returns spectrogram and label of given index
        Inputs:
            index   -> Index of spectrogram/label to be returned [int]
        Outputs:
            output_spec     -> Spectrogram of given index [tensor]
            label           -> Label of given index [tensor]
        '''
        data_dir = self.data_dirs[index]    # get the subdirectory for the given index
        label_path = os.path.join(self.path, data_dir, self.label_filename) # construct the full path to the label file
        spec_path = os.path.join(self.path, data_dir, self.spec_filename)   # construct the full path to the spectrogram file

        if self.transform:
            spec, input_time, input_wavelength, output_spec, output_time, output_wavelength = self.transform(spec_path)
            # output_spec = torch.tensor(output_spec, dtype=torch.float64)
        else:
            output_spec = torch.tensor(pd.read_csv(spec_path, header=None, engine='python').values, dtype=torch.half).unsqueeze(0)

        if self.target_transform:
            label = self.target_transform(label_path)
            # label = torch.from_numpy(label)
        else:
            label = torch.tensor(pd.read_csv(label_path, header=None, engine='python').values).unsqueeze(0)

        # create a spectrogram with 3 identical channels
        output_spec = output_spec.unsqueeze(0)  # add another dimension to the tensor
        output_spec = output_spec.repeat(3,1,1) # repeat the spectrogram 3 times (3,h,w)

        # ensure correct output data type
        if not isinstance(output_spec, torch.Tensor):
            output_spec = torch.tensor(output_spec)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        return output_spec, label

'''
ResampleSpectrogram()
Transform Class that resamples spectrograms to use the same axis and size
'''
class ResampleSpectrogram(object):

    def __init__(self, num_delays_out, timestep_out, num_frequencies_out, start_frequency_out, end_frequency_out):
        '''
        Inputs:
            num_delays_out          -> Number of delay values the resampled spectrogram will have [int]
            timestep_out            -> Length of time between delays [fs] [int]
            num_frequencies_out     -> Number of frequency values the resampled spectrogram will have [int]
            start_frequency_out     -> Lowest frequency value of the resampled spectrogram [nm] [int]
            end_frequency_out       -> Highest frequency value of the resampled spectrogram [nm] [int]
        '''
        output_number_rows = num_delays_out
        output_time_step = timestep_out
        output_number_cols = num_frequencies_out
        output_start_frequency = start_frequency_out   # the first element of the frequency output axis
        output_stop_frequency = end_frequency_out    # the last element of the frequency output axis 
        output_start_time = -int(output_number_rows/2) * output_time_step     # calculate time at which the output time axis starts
        output_end_time = output_start_time + (output_time_step * output_number_rows) - output_time_step    # calculate the last element of the output time axis 

        ######################## Used here to save time later
        ## Define output axis ##
        ########################
        self.output_time = np.linspace(output_start_time, output_end_time, output_number_rows )  # create array that corresponds to the output time axis
        self.output_frequency = np.linspace(output_start_frequency, output_stop_frequency, output_number_cols)    # create array that corresponds tot the output wavelength axis
 

    def __call__(self, path):
        '''
        Takes path of spectrogram and resamples it to the configured size and range of time/wavelength 
        Inputs:
            path    -> Path to spectrogram [string]
        Outputs:
            spectrogram             -> Original (not resampled) spectrogram [tensor]
            input_time              -> Original (not resampled) time axis [numpy array]
            input_wavelength        -> Original (not resampled) wavelength axis [numpy array]
            output_spectrogram      -> Resampled spectrogram [tensor]
            self.output_time        -> Resampled time axis [numpy array]
            self.output_wavelength  -> Resampled frequency axis [numpy array]
        '''
        # Constants 
        NUM_HEADER_ELEMENTS = 5
        SPEED_OF_LIGHT = 299792458

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
                # print(f'Error: Number of Header Elements != {NUM_HEADER_ELEMENTS}')
                return
            else:
                # fist line has 5 Elements -> write into header
                header = first_line.split()
                num_rows_skipped = 1 # header is in first row
                # print("Header is in 1 rows")    # for debugging
        else:
            # first 2 lines have 5 Elements in total -> write into header
            header = first_line.split() + second_line.split()
            num_rows_skipped = 2 # header is in first 2 rows
            # print("Header is in 2 rows")    # for debugging
        
        input_number_rows = int(header[0]) # delay points
        input_number_cols = int(header[1]) # wavelength points
        input_time_step = float(header[2]) # [fs]
        input_wavelength_step = float(header[3]) # [nm]
        input_center_wavelength = float(header[4]) # [nm]
        
        ######################
        ## Read Spectrogram ##
        ######################
        spectrogram_df = pd.read_csv(path, sep='\\s+', skiprows=num_rows_skipped, header=None, engine='python')
        spectrogram = spectrogram_df.to_numpy()     # convert to numpy array 
        spectrogram = torch.from_numpy(spectrogram)     # convert to tensor

        #######################
        ## Define input axis ##
        #######################
        # calculate input time axis
        input_start_time = -int(input_number_rows / 2) * input_time_step    # calculate time at which the input time axis starts
        input_end_time = input_start_time + (input_time_step * input_number_rows) - input_time_step    # calculate the last element of the input time axis
        input_time = np.linspace(input_start_time, input_end_time, input_number_rows)   # create array that corresponds to the input time axis
        
        # calculate input wavelength axis
        input_start_wavelength = input_center_wavelength - (input_number_cols/2)*input_wavelength_step      # calculate the first element of the wavelength input axis
        input_stop_wavelength = input_center_wavelength + (input_number_cols/2)*input_wavelength_step       # calculate the last element of the wavelength input axis
        input_wavelength = np.linspace(input_start_wavelength, input_stop_wavelength, input_number_cols)    # create array that corresponds tot the input wavelength axis
        
        ##########################
        ## Convert to frequency ##
        ##########################
        input_freq = (SPEED_OF_LIGHT * 1e9) / input_wavelength # convert wavelenght [nm] to frequency [Hz]
        input_freq = input_freq[::-1]   # ensure increasing order of frequency
        
        ##############################
        ## Resample frequency axis ##
        ##############################
        interpolate_frequency = interp1d(input_freq, spectrogram, axis=1, kind='linear', bounds_error=False, fill_value=0) 
        output_spectrogram = interpolate_frequency(self.output_frequency)
        
        ########################
        ## Resample time axis ##
        ######################## 
        interpolate_time = interp1d(input_time, output_spectrogram, axis=0, kind='linear', bounds_error=False, fill_value=0)
        output_spectrogram = interpolate_time(self.output_time)
        output_spectrogram = torch.from_numpy(output_spectrogram)  # convert to tensor
        spectrogram = spectrogram

        return spectrogram, input_time, input_wavelength, output_spectrogram, self.output_time, self.output_frequency

'''
ReadLabelFromEs()
Read labels (real and imag part) from Es.dat
'''
class ReadLabelFromEs(object):
    def __init__(self, number_elements):
        '''
        Inputs:
            number_elements     -> Number of elements in the intensity and phase array each [int]
        '''
        self.number_elements = number_elements

    def __call__(self, path):    
        '''
        Read Ek.dat file and place columns in arrays
        Inputs:
            Path -> Path to Ek.dat [string]
        Outputs:
            label -> List of arrays containing real and imag part of time signal [tensor]
            [
                time_axis   -> Array containing time axis of time signal [numpy array]
                intensity   -> Array containing intensity of time signal (squared amplitude) [numpy array]
                phase       -> Array containing phase of time signal [numpy array]
                real_part   -> Array containing real part of time signal [numpy array]
                imag_part   -> Array containing imaginary part of time signal [numpy array]
            ]
        '''
        # read the dataframe
        dataframe = pd.read_csv(path,sep='  ', decimal=",", header=None, engine='python')     # sep needs to be 2 spaces
        
        TimeDomainSignal = TimeDomain(time_axis = dataframe[0].to_numpy(),
                                      intensity = dataframe[1].to_numpy(),
                                      phase = dataframe[2].to_numpy(), 
                                      real = dataframe[3].to_numpy(),
                                      imag = dataframe[4].to_numpy())

        # Resample to fit correct number of elements
        # TimeDomainSignal.intensity = TimeDomainSignal.intensity.reshape(self.number_elements,2).mean(axis=1)
        # TimeDomainSignal.phase = TimeDomainSignal.phase.reshape(self.number_elements,2).mean(axis=1)
        original_indicies = np.linspace(0, len(TimeDomainSignal.real) - 1, num=len(TimeDomainSignal.real))
        new_indicies = np.linspace(0, len(TimeDomainSignal.real) - 1, num=self.number_elements)
        interpolation_func_real = interp1d(original_indicies, TimeDomainSignal.real, kind='linear')
        interpolation_func_imag = interp1d(original_indicies, TimeDomainSignal.imag, kind='linear')
        TimeDomainSignal.real = interpolation_func_real(new_indicies)
        TimeDomainSignal.imag = interpolation_func_imag(new_indicies)
        # TimeDomainSignal.real = TimeDomainSignal.real.reshape(self.number_elements,2).mean(axis=1)
        # TimeDomainSignal.imag = TimeDomainSignal.imag.reshape(self.number_elements,2).mean(axis=1)

        # label = np.concatenate( (TimeDomainSignal.intensity, TimeDomainSignal.phase), axis=0)
        label = np.concatenate( (TimeDomainSignal.real, TimeDomainSignal.imag), axis=0)
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
            number_elements     -> Number of elements in the intensity and phase array each [int]
        '''
        self.number_elements = number_elements
        # get the center index of each half
        self.index_center = number_elements // 2

    def __call__(self, label):    
        '''
        Read Ek.dat file and place columns in arrays
        Inputs:
            label -> List of arrays containing real and imag part of time signal [tensor]
        Outputs:
            output_label -> List of arrays containing real and imag part of time signal, without ambiguities [tensor]
        '''
        real_part = label[:self.number_elements]
        imag_part = label[self.number_elements:]

        # calculate the intensity of the signal
        intensity = real_part**2 + imag_part**2

        # get the index of the highest intensity value and calculate the offset
        index_peak = torch.argmax(intensity)
        offset = self.index_center - index_peak

        # shift the real and imaginary parts to center the peak
        real_part = torch.roll(real_part, offset.item() )
        imag_part = torch.roll(imag_part, offset.item() )
        
        # Calculate the mean of the pulse before and after the center index
        mean_first_half = torch.mean(intensity[:self.index_center])
        mean_second_half = torch.mean(intensity[self.index_center:])

        # if the mean after the intensity peak is higher, mirror real and imaginary part
        if mean_second_half > mean_first_half:
            real_part = torch.flip(real_part, dims=[0])
            imag_part = torch.flip(imag_part, dims=[0])
        
        output_label = torch.cat([real_part, imag_part])
        return output_label

class Scaler(object):
    def __init__(self, number_elements, max_real, max_imag):
        '''
        Inputs:
            number_elements -> Number of elements in each the real and imaginary part of the array [int]
            max_intensity   -> >= maximum real part in dataset [float]
            max_phase       -> >= maximum imag part in dataset [float]
        '''
        self.number_elements = number_elements
        self.max_real = max_real
        self.max_imag = max_imag

    def scale(self, label):
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            label -> List of arrays containing real and imaginary part of time signal [tensor]
        Outputs:
            scaled_label -> List of arrays containing real and imaginary part of the time signal scaled to [-1,1] [tensor]
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
            scaled_label -> List of arrays containing the real and imaginary part scaled to [-1,1] [tensor]
        Outputs:
            label -> List of arrays containing the real and imaginary part of time signal [tensor]
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
ScaleLabel()
Scales Labels to range [-1,1]
'''
class ScaleLabel(object):
    def __init__(self, max_intensity, max_phase):
        '''
        Inputs:
            max_intensity   -> >= maximum intesity in dataset [float]
            max_phase       -> >= maximum phase in dataset [float]
        '''
        self.max_intensity = max_intensity
        self.max_phase = max_phase

    def __call__(self, label):    
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase [tensor]
        Outputs:
            scaled_label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
                scaled to [-1,1] [tensor]
        '''
        length_label = len(label)
        half_size = int(length_label //2)
        intensity = label[:half_size]  # First half -> intensity
        phase = label[half_size:]      # Second half -> phase
        intensity_scaled = intensity / self.max_intensity
        phase_scaled = phase / self.max_phase

        # Concatenate the two halves back together
        scaled_label = torch.cat((intensity_scaled, phase_scaled), dim=0)

        return scaled_label

'''
UnscaleLabel()
Restores original scale of labels
'''
class UnscaleLabel(object):
    def __init__(self, max_intensity, max_phase):
        '''
        Inputs:
            max_intensity   -> >= maximum intesity in dataset (needs to be the same as in ScaleLabel) [float]
            max_phase       -> >= maximum phase in dataset (needst to be the same as in ScaleLabel) [float]
        '''
        self.max_intensity = max_intensity
        self.max_phase = max_phase

    def __call__(self, scaled_label):    
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            scaled_label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
                scaled to [-1,1] [tensor]
        Outputs:
            label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase [tensor]
        '''
        length_label = len(scaled_label)
        half_size = int(length_label //2)
        intensity_scaled = scaled_label[:half_size]  # First half -> intensity
        phase_scaled = scaled_label[half_size:]      # Second half -> phase
        intensity = intensity_scaled * self.max_intensity
        phase = phase_scaled * self.max_phase

        # Concatenate the two halves back together
        label = torch.cat((intensity, phase), dim=0)

        return label

'''
UnscaleLabel()
Restores original scale of labels
'''
class UnscaleLabel(object):
    def __init__(self, max_intensity, max_phase):
        '''
        Inputs:
            max_intensity   -> >= maximum intesity in dataset (needs to be the same as in ScaleLabel) [float]
            max_phase       -> >= maximum phase in dataset (needst to be the same as in ScaleLabel) [float]
        '''
        self.max_intensity = max_intensity
        self.max_phase = max_phase

    def __call__(self, scaled_label):    
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            scaled_label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
                scaled to [-1,1] [tensor]
        Outputs:
            label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase [tensor]
        '''
        length_label = len(scaled_label)
        half_size = int(length_label //2)
        intensity_scaled = scaled_label[:half_size]  # First half -> intensity
        phase_scaled = scaled_label[half_size:]      # Second half -> phase
        intensity = intensity_scaled * self.max_intensity
        phase = phase_scaled * self.max_phase

        # Concatenate the two halves back together
        label = torch.cat((intensity, phase), dim=0)

        return label

'''
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

        self.penalty_factor = penalty_factor
        self.threshold = threshold

    def forward(self, predictions, labels):
        # print(f"prediction size = {predictions}")
        # print(f"label size = {labels}")
        
        # get the number of batches, as well as the shape of the labels
        batch_size, num_elements = labels.shape
        # get half of elements
        half_size = num_elements // 2

        # get real and imaginary parts of labels and predictions
        labels_real = labels[:, :half_size]
        labels_imag = labels[:, half_size:]

        predictions_real = predictions[:, :half_size]
        predictions_imag = predictions[:, half_size:]
        
        # initialize loss
        loss = 0.0
        # Loop over each batch
        for i in range(batch_size):
            # Create masks for all absolute values higher than the threshold
            mask_real_threshold = abs(labels_real[i]) > self.threshold
            mask_imag_threshold = abs(labels_imag[i]) > self.threshold
            
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
            mse_real = (predictions_real[i] - labels_real[i]) ** 2
            mse_imag = (predictions_imag[i] - labels_imag[i]) ** 2

            # Apply penalty for values before the first significant index and after the last
            mse_real[:first_significant_idx_real] *= self.penalty_factor
            mse_real[last_significant_idx_real + 1:] *= self.penalty_factor
            mse_imag[:first_significant_idx_imag] *= self.penalty_factor
            mse_imag[last_significant_idx_imag + 1:] *= self.penalty_factor
            
            # Add to total loss
            loss += mse_real.mean() + mse_imag.mean()
        # devide by batch size 
        loss = loss / batch_size

        return loss
