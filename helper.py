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
            num_outputs     -> Number of outputs of the DenseNet
        '''
        super(CustomDenseNet, self).__init__()
        # Load pretrained DenseNet
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # Get the number of features before the last layer
        num_features = self.densenet.classifier.in_features
        # Create a Layer with the number of features before the last layer and 256 outputs (2 arrays of 128 Elements)
        self.densenet.classifier = nn.Linear(num_features, num_outputs)
        self.num_outputs = num_outputs

    def forward(self, spectrogram):
        '''
        Forward pass through the DenseNet
        Input:
            spectrogram     -> input spectrogram
        Output:
            x   -> predicted output
        '''
        half_size = int(self.num_outputs //2)
        # get the output of the densenet
        x = self.densenet(spectrogram)
        # use tanh activation function to scale the output to [-1, 1] and then scale it (intensity)
        x = torch.tanh(x)
        # use tanh activation function to scale the output to [-1, 1] and then scale it (intensity)
        # intensity = torch.tanh(x[:half_size])
        # use sigmoid activation function to scale the output to [0, 1] and then scale it
        # phase = torch.sigmoid(x[half_size:])

        # x = torch.cat((intensity, phase), dim=0)

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
    def __init__(self, path, label_filename, spec_filename, transform=None, target_transform=None):
        '''
        Inputs:
            path                -> root directory containing all data subdirectories
            label_filename      -> file name in which labels are stored
            spec_filename       -> file name in which spectrograms are stored
            transform           -> transform used on spectrograms
            target_transform    -> transforms used on labels
        '''
        self.path = path    # root directory containing all data subdirectories
        self.label_filename = label_filename      # file name in which labels are stored
        self.spec_filename = spec_filename        # file name in which spectrograms are stored
        self.transform = transform              # transform used on spectrograms
        self.target_transform = target_transform    # transforms used on labels

        self.data_dirs = os.listdir(self.path)  # list all subdirectories in the root directory
        
    def __len__(self):
        '''
        Returns the number of data subdirectories (number of spectrograms)
        '''
        return len(self.data_dirs)              # return the number of data subdirectories

    def __getitem__(self, index):
        '''
        Returns spectrogram and label of given index
        Inputs:
            index   -> Index of spectrogram/label to be returned
        Outputs:
            output_spec     -> Spectrogram of given index
            label           -> Label of given index
        '''
        data_dir = self.data_dirs[index]    # get the subdirectory for the given index
        label_path = os.path.join(self.path, data_dir, self.label_filename) # construct the full path to the label file
        spec_path = os.path.join(self.path, data_dir, self.spec_filename)   # construct the full path to the spectrogram file

        if self.transform:
            spec, input_time, input_wavelength, output_spec, output_time, output_wavelength = self.transform(spec_path)
            output_spec = torch.tensor(output_spec, dtype=torch.float64)
        else:
            output_spec = torch.tensor(pd.read_csv(spec_path, header=None, engine='python').values, dtype=torch.float64).unsqueeze(0)

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

    def __init__(self, num_delays_out, timestep_out, num_wavelength_out, start_wavelength_out, end_wavelength_out):
        '''
        Inputs:
            num_delays_out          -> Number of delay values the resampled spectrogram will have
            timestep_out            -> Length of time between delays [fs]
            num_wavelength_out      -> Number of wavelength values the resampled spectrogram will have
            start_wavelength_out    -> Lowest wavelength value of the resampled spectrogram [nm]
            end_wavelength_out      -> Highest wavelength value of the resampled spectrogram [nm]
        '''
        output_number_rows = num_delays_out
        output_time_step = timestep_out
        output_number_cols = num_wavelength_out
        output_start_wavelength = start_wavelength_out   # the first element of the wavelength output axis
        output_stop_wavelength = end_wavelength_out    # the last element of the wavelength output axis 
        output_start_time = -int(output_number_rows/2) * output_time_step     # calculate time at which the output time axis starts
        output_end_time = output_start_time + (output_time_step * output_number_rows) - output_time_step    # calculate the last element of the output time axis 

        ######################## Used here to save time later
        ## Define output axis ##
        ########################
        self.output_time = np.linspace(output_start_time, output_end_time, output_number_rows )  # create array that corresponds to the output time axis
        self.output_wavelength = np.linspace(output_start_wavelength, output_stop_wavelength, output_number_cols)    # create array that corresponds tot the output wavelength axis
 

    def __call__(self, path):
        '''
        Takes path of spectrogram and resamples it to the configured size and range of time/wavelength 
        Inputs:
            path    -> Path to spectrogram
        Outputs:
            spectrogram             -> Original (not resampled) spectrogram
            input_time              -> Original (not resampled) time axis
            input_wavelength        -> Original (not resampled) wavelength axis
            output_spectrogram      -> Resampled spectrogram
            self.output_time        -> Resampled time axis
            self.output_wavelength  -> Resampled wavelenght axis
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
        
        ##############################
        ## Resample wavelength axis ##
        ##############################
        interpolate_wavelength = interp1d(input_wavelength, spectrogram, axis=1, kind='linear', bounds_error=False, fill_value=0) 
        output_spectrogram = interpolate_wavelength(self.output_wavelength)
        
        ########################
        ## Resample time axis ##
        ######################## 
        interpolate_time = interp1d(input_time, output_spectrogram, axis=0, kind='linear', bounds_error=False, fill_value=0)
        output_spectrogram = interpolate_time(self.output_time)

        return spectrogram, input_time, input_wavelength, output_spectrogram, self.output_time, self.output_wavelength
 
'''
ReadLabelFromEs()
Read labels (intensity and phase) from Es.dat
'''
class ReadLabelFromEs(object):
    def __init__(self, number_elements):
        '''
        Inputs:
            number_elements     -> Number of elements in the intensity and phase array each
        '''
        self.number_elements = number_elements

    def __call__(self, path):    
        '''
        Read Ek.dat file and place columns in arrays
        Inputs:
            Path -> Path to Ek.dat 
        Outputs:
            label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
            [
                time_axis   -> Array containing time axis of time signal
                intensity   -> Array containing intensity of time signal (squared amplitude)
                phase       -> Array containing phase of time signal
                real_part   -> Array containing real part of time signal
                imag_part   -> Array containing imaginary part of time signal
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

        TimeDomainSignal.real = TimeDomainSignal.real.reshape(self.number_elements,2).mean(axis=1)
        TimeDomainSignal.imag = TimeDomainSignal.imag.reshape(self.number_elements,2).mean(axis=1)

        # label = np.concatenate( (TimeDomainSignal.intensity, TimeDomainSignal.phase), axis=0)
        label = np.concatenate( (TimeDomainSignal.real, TimeDomainSignal.imag), axis=0)
        label = torch.from_numpy(label).float()
        return label

class RemoveAbsolutePhaseShift(object):
    def __init__(self):
        pass
    def __call__(self,label):
        length_label = len(label)
        half_size = int(length_label //2)
        intensity = label[:half_size]  # First half -> intensity
        phase = label[half_size:]      # Second half -> phase
        phase_corrected_range = np.mod(phase, 2*np.pi)
        new_label = torch.cat((intensity, phase_corrected_range), dim=0)
        return new_label


'''
ScaleLabel()
Scales Labels to range [-1,1]
'''
class ScaleLabel(object):
    def __init__(self, max_intensity, max_phase):
        '''
        Inputs:
            max_intensity   -> >= maximum intesity in dataset
            max_phase       -> >= maximum phase in dataset
        '''
        self.max_intensity = max_intensity
        self.max_phase = max_phase

    def __call__(self, label):    
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
        Outputs:
            scaled_label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
                scaled to [-1,1]
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
            max_intensity   -> >= maximum intesity in dataset (needs to be the same as in ScaleLabel)
            max_phase       -> >= maximum phase in dataset (needst to be the same as in ScaleLabel)
        '''
        self.max_intensity = max_intensity
        self.max_phase = max_phase

    def __call__(self, scaled_label):    
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            scaled_label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
                scaled to [-1,1]
        Outputs:
            label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
        '''
        length_label = len(scaled_label)
        half_size = int(length_label //2)
        intensity_scaled = scaled_label[:half_size]  # First half -> intensity
        phase_scaled = scaled_label[half_size:]      # Second half -> phase
        intensity = torch.from_numpy(intensity_scaled * self.max_intensity).float()
        phase = torch.from_numpy(phase_scaled * self.max_phase).float()

        # Concatenate the two halves back together
        label = torch.cat((intensity, phase), dim=0)

        return label
