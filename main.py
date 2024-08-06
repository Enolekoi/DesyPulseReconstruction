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
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt

import random
import logging

# Classes, methods and functions from different files
import helper
import visualize as vis
import config

'''
Variables and settings
'''
# Paths
LogDirectory = "./logs/"
LogName = "training_"
TrainingLossPlotName = "training_loss_"

Path = "/mnt/data/desy/frog_simulated/grid_256/"
SpecFilename = "as.dat"
LabelFilename = "Es.dat"

# Constants
OUTPUT_SIZE = 256
BATCH_SIZE = 10
NUM_EPOCHS = 2
LEARNING_RATE = 0.0001

OUTPUT_NUM_DELAYS = 512 
OUTPUT_NUM_WAVELENGTH = 512 
OUTPUT_TIMESTEP = 5    # [fs]
OUTPUT_START_WAVELENGTH = 350   # [nm]
OUTPUT_END_WAVELENGTH = 550     # [nm]

# get the correct filepaths of all files
log_filepath, loss_plot_filepath = config.getLogFilepath(
        directory=LogDirectory,
        log_base_filename=LogName,
        loss_plot_base_filename=TrainingLossPlotName
        )


# Logger Settings
logging.basicConfig(
        level=logging.INFO,
        style="{",
        format="{asctime} - {name} - {funcName} - {levelname}: {message}",
        datefmt='%d-%m-%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
            ]
)
logger = logging.getLogger(__name__)

# Log some information
logger.info(f"Size of Output Tensor: {2*OUTPUT_SIZE} Elements")
logger.info(f"Batch Size: {BATCH_SIZE} Elements")
logger.info(f"Number of Epochs: {NUM_EPOCHS}")
logger.info(f"Learning Rate: {LEARNING_RATE}")

# Transforms
spec_transform = helper.ResampleSpectrogram(OUTPUT_NUM_DELAYS, OUTPUT_TIMESTEP, OUTPUT_NUM_WAVELENGTH, OUTPUT_START_WAVELENGTH, OUTPUT_END_WAVELENGTH)
label_transform = helper.ReadLabelFromEs()

# Define device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Using {device} as device!')
logger.info(f"Device used (cuda/cpu): {device}")
if device == 'cuda':
    torch.cuda.empty_cache()

'''
Load Model
'''
logger.info("Loading Model...")
# Load custom DenseNet
model = helper.CustomDenseNet(output_scale=40, num_outputs=2*OUTPUT_SIZE)
model.float()
model.to(device)
model.eval()

logger.info("Loading Model finished!")
# print('Loading Model finished!')

'''
Load Data
'''
# print('Loading Data...')
logger.info("Loading Data...")
data = helper.SimulatedDataset(path=Path,
                               label_filename=LabelFilename,
                               spec_filename=SpecFilename,
                               transform=spec_transform,
                               target_transform=label_transform)
################
## Split Data ##
################
length_dataset = len(data)  # get length of data
# print(f'Size of Dataset: {length_dataset}')
logger.info(f"Size of dataset: {length_dataset}")

# get ratios
train_size = int(0.8 * length_dataset)  # amount of training data (80%)
validation_size = int(0.1 * length_dataset)     # amount of validation data (10%)
test_size = length_dataset - train_size - validation_size   # amount of test data (10%)

logger.info(f"Size of training data: {train_size}")
logger.info(f"Size of validation data: {validation_size}")
logger.info(f"Size of test data: {test_size}")

# split 
train_data, validation_data, test_data = random_split(data, [train_size, validation_size, test_size])   # split data

# Data Loaders
train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size = BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle=False)

# print('Loading Data finished')
logger.info("Loading Data finished!")

'''
Training
'''
# print('Starting Training...')
logger.info("Starting Training...")
########################
## loss and optimizer ##
########################
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_values = []

num_total_steps = len(train_loader)
for epoch in range(NUM_EPOCHS):     # iterate over epochs
    for i, (spectrograms, labels) in enumerate(train_loader): # iterate over spectrograms and labels of train_loader
        # print(spectrograms.shape)
        # print(type(spectrograms))
        spectrograms = spectrograms.float()
        # send spectrogram and label data to selected device
        spectrograms = spectrograms.to(device)
        labels = labels.float().to(device)
        
        # Forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print information (every 100 steps)
        if (i+1) % 10 == 0:
            # print(f'Epoch {epoch+1} / {NUM_EPOCHS}, Step {i+1} / {num_total_steps}, Loss = {loss.item():.10f}')
            logger.info(f"Epoch {epoch+1} / {NUM_EPOCHS}, Step {i+1} / {num_total_steps}, Loss = {loss.item():.10f}")
        # Write loss into array
        loss_values.append(loss.item())
vis.save_plot_training_loss(loss_values, config.loss_plot_filepath)
logger.info("Saved plot of training loss!")
# print('Training finished')
logger.info("Training finished!")

# Visualize training
'''
validation
'''
# print('Starting Validation...')
logger.info("Starting Validation...")
model.eval()
with torch.no_grad():
    validation_sample = random.choice(validation_data)
    spectrogram, label = validation_sample
    spectrogram = spectrogram.float().unsqueeze(0).to(device)
    label = label.float().to(device)

    prediction = model(spectrogram).cpu().numpy().flatten()
    
    original_label = label.cpu().numpy().flatten()
    
    vis.compareTimeDomain("./random_prediction.png", original_label, prediction)
    # vis.visualize(spectrogram, original_label, prediciton)
logger.info("Validation finished!")

for handler in logger.handlers:
    handler.flush()
    handler.close()
