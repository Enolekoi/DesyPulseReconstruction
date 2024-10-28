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
CustomDenseNetTBDrms()
Custom DenseNet class for predictiong the time bandwith product from spectrogram
'''
class CustomDenseNetTBDrms(nn.Module):
    def __init__(self):
        '''
        Inputs:
            num_outputs     -> Number of outputs of the DenseNet [int]
        '''
        super(CustomDenseNetTBDrms, self).__init__()
        # Load pretrained DenseNet
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # self.densenet = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        # Get the number of features before the last layer
        num_features = self.densenet.classifier.in_features
        # Create a Layer with the number of features before the last layer and 256 outputs (2 arrays of 128 Elements)
        self.densenet.classifier = nn.Linear(num_features, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 1)
        # self.densenet.classifier = nn.Linear(num_features, num_outputs)

        # initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu') # useful before relu
        nn.init.zeros_(self.fc1.bias)  # Initialize biases to zero
        nn.init.xavier_normal_(self.fc2.weight) # useful before tanh
        nn.init.zeros_(self.fc2.bias)  # Initialize biases to zero

    def forward(self, spectrogram):
        '''
        Description:
            Forward pass through the DenseNet
        Input:
            spectrogram     -> [tensor] input spectrogram
        Output:
            x               -> [tensor] predicted output
        '''
        # half_size = int(self.num_outputs //2)
        # get the output of the densenet
        x = self.densenet(spectrogram)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x = torch.relu(x)
        
        # use sigmoid activation function for output to be between 0 and 1 -> then scale with 20 for larger TBDrms values
        x = torch.sigmoid(x) * 15
        return x

'''
LoadDatasetTBDrms()
Custom Dataloader Class
'''
class LoadDatasetTBDrms(Dataset):
    def __init__(self, path, tbd_filename, spec_filename, transform=None, target_transform=None):
        '''
        Inputs:
            path                -> root directory containing all data subdirectories [string]
            tbd_filename        -> file name in which tbd_rms values are stored [string]
            spec_filename       -> file name in which spectrograms are stored [string] 
            transform           -> transform used on spectrograms 
            target_transform    -> transforms used on labels
        '''
        self.path = path    # root directory containing all data subdirectories
        self.spec_filename = spec_filename        # file name in which spectrograms are stored
        self.target_transform = target_transform    # transforms used on labels
        self.transform = transform              # transform used on spectrograms
        self.target_transform = target_transform    # transforms used on labels

        self.data_dirs = os.listdir(self.path)  # list all subdirectories in the root directory
        
        # Load the TBDrms file
        tbdrms_data = pd.read_csv(tbd_filename)
        print(tbdrms_data.shape)
        tbdrms_data['index'] = tbdrms_data['Directory'].str[1].astype(int)
        tbdrms_data = tbdrms_data.sort_values('index')
        tbdrms_data = tbdrms_data.drop('index', axis=1)
        self.tbdrms_values = tbdrms_data['TBDrms']

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
        INDEX_TBD_COLUMN = 3
        data_dir = self.data_dirs[index]    # get the subdirectory for the given index
        spec_path = os.path.join(self.path, data_dir, self.spec_filename)   # construct the full path to the spectrogram file
        label = self.tbdrms_values.iloc[index].squeeze(0)

        if self.transform:
            spec, input_time, input_wavelength, output_spec, output_time, output_wavelength = self.transform(spec_path)
            # output_spec = torch.tensor(output_spec, dtype=torch.float64)
        else:
            output_spec = torch.tensor(pd.read_csv(spec_path, header=None, engine='python').values, dtype=torch.half).unsqueeze(0)

        # create a spectrogram with 3 identical channels
        # output_spec = output_spec.unsqueeze(0)  # add another dimension to the tensor
        # output_spec = output_spec.repeat(3,1,1) # repeat the spectrogram 3 times (3,h,w)

        # ensure correct output data type
        if not isinstance(output_spec, torch.Tensor):
            output_spec = torch.tensor(output_spec)

        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)

        return output_spec, label


