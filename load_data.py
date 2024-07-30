import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

class SimulatedDataset(Dataset):
    def __init__(self, path, label_filename, spec_filename, transform=None, target_transform=None):
        self.path = path    # root directory containing all data subdirectories
        self.label_filename = label_filename      # file name in which labels are stored
        self.spec_filename = spec_filename        # file name in which spectrograms are stored
        self.transform = transform              # transform used on spectrograms
        self.target_transform = target_transform    # transforms used on labels

        self.data_dirs = os.listdir(self.path)  # list all subdirectories in the root directory
        
    def __len__(self):
        return len(self.data_dirs)              # return the number of data subdirectories

    def __getitem__(self, index):
        data_dir = self.data_dirs[index]    # get the subdirectory for the given index
        label_path = os.path.join(self.path, data_dir, self.label_filename) # construct the full path to the label file
        spec_path = os.path.join(self.path, data_dir, self.spec_filename)   # construct the full path to the spectrogram file

        if self.transform:
            spec, input_time, input_wavelength, output_spec, output_time, output_wavelength = self.transform(spec_path)
            print("here spec trans")
        else:
            output_spec = torch.tensor(pd.read_csv(spec_path, header=None).values).unsqueeze(0)

        if self.target_transform:
            label = self.target_transform(label_path)
            print("here label trans")
        else:
            label = torch.tensor(pd.read_csv(label_path, header=None).values).unsqueeze(0)

        print(output_spec.shape)
        print(label.shape)

        return output_spec, label
