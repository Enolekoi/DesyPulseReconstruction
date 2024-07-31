'''
Libraries and classes/methods
'''
# Libraries used in this file
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

# Classes, methods and functions from different files
import helper
import visualize as vis

'''
Variables and settings
'''
# Define device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Paths
Path = "/mnt/data/desy/frog_simulated/grid_256/"
SpecFilename = "as.dat"
LabelFilename = "Es.dat"

# Constants
OUTPUT_NUM_DELAYS = 512 
OUTPUT_NUM_WAVELENGTH = 512 
OUTPUT_TIMESTEP = 5    # [fs]
OUTPUT_START_WAVELENGTH = 350   # [nm]
OUTPUT_END_WAVELENGTH = 550     # [nm]

output_size = 128
batch_size = 100
num_epochs = 2
learning_rate = 0.001

# Transforms
spec_transform = helper.ResampleSpectrogram(OUTPUT_NUM_DELAYS, OUTPUT_TIMESTEP, OUTPUT_NUM_WAVELENGTH, OUTPUT_START_WAVELENGTH, OUTPUT_END_WAVELENGTH)
label_transform = helper.ReadLabelFromEs()

'''
Load Model
'''

# Load pretrained DenseNet
model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
# Get the number of features before the last layer
num_features = model.classifier.in_features
# Create a Layer with the number of features before the last layer and 256 outputs (2 arrays of 128 Elements)
model.classifier = nn.Linear(num_features, 256)

model.to(device)
model.eval()

'''
Load Data
'''
data = helper.SimulatedDataset(path=Path,
                               label_filename=LabelFilename,
                               spec_filename=SpecFilename,
                               transform=spec_transform,
                               target_transform=label_transform)
################
## Split Data ##
################
length_dataset = len(data)  # get length of data
# get ratios
train_size = int(0.8 * length_dataset)  # amount of training data (80%)
validation_size = int(0.1 * length_dataset)     # amount of validation data (10%)
test_size = length_dataset - train_size - validation_size   # amount of test data (10%)
# split 
train_data, validation_data, test_data = random_split(data, [train_size, validation_size, test_size])   # split data

# Data Loaders
train_loader = DataLoader(train_data, batch_size = batch_size, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle=False)
train_loader = DataLoader(test_data, batch_size = batch_size, shuffle=False)

'''
Training
'''
########################
## loss and optimizer ##
########################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):     # iterate over epochs
    for i, (spectrograms, labels) in enumerate(train_loader): # iterate over spectrograms and labels of train_loader
        print(spectrograms.shape)
        # send spectrogram and label data to selected device
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(spectrograms)
        output_array = outputs.detach().numpy()
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print information (every 100 steps)
        if (i+1) % 100 == 0:
            print(f'Epoch {epoch+1} / {num_epochs}, Step {i+1} / {n_total_steps}, Loss = {loss.item():.4f}')
'''
validation
'''
# visualize random prediction
image, label = data[0]
prediction = np.zeros(1)
vis.visualize(image, label, prediction)
