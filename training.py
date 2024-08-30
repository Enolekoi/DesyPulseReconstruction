'''
Libraries and classes/methods
'''
# Libraries used in this file
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt

import random
import logging

from sklearn.model_selection import KFold
# Classes, methods and functions from different files
import helper
import visualize as vis
import config

'''
Variables and settings
'''
# Logger Settings
logging.basicConfig(
        level=logging.INFO,
        style="{",
        format="{asctime} - {name} - {funcName} - {levelname}: {message}",
        datefmt='%d-%m-%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(config.log_filepath),
            logging.StreamHandler()
            ]
)
logger = logging.getLogger(__name__)

# Log some information
logger.info(f"Writing into log file: {config.log_filepath}")
logger.info(f"Dataset used: {config.Path}")
logger.info(f"Noise level used: {config.SpecFilename}")
logger.info(f"Size of output tensor: {2*config.OUTPUT_SIZE} elements")
logger.info(f"Batch size: {config.BATCH_SIZE} elements")
logger.info(f"Number of epochs: {config.NUM_EPOCHS}")
logger.info(f"Initial learning rate: {config.LEARNING_RATE}")

# Transforms
spec_transform = helper.ResampleSpectrogram(config.OUTPUT_NUM_DELAYS, config.OUTPUT_TIMESTEP, config.OUTPUT_NUM_WAVELENGTH, config.OUTPUT_START_WAVELENGTH, config.OUTPUT_END_WAVELENGTH)
label_reader = helper.ReadLabelFromEs(config.OUTPUT_SIZE)
label_phase_correction = helper.RemoveAbsolutePhaseShift()
label_scaler = helper.ScaleLabel(max_intensity=config.MAX_INTENSITY, max_phase=config.MAX_PHASE)
label_transform = transforms.Compose([label_reader, label_phase_correction, label_scaler])

label_unscaler = helper.UnscaleLabel(max_intensity=config.MAX_INTENSITY, max_phase=config.MAX_PHASE)

# Define device used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device used (cuda/cpu): {device}")
if device == 'cuda':
    torch.cuda.empty_cache()

'''
Load Model
'''
logger.info("Loading Model...")
# Load custom DenseNet
model = helper.CustomDenseNet(
    num_outputs=2*config.OUTPUT_SIZE
    )

model.float()
model.to(device)
model.eval()

logger.info("Loading Model finished!")

'''
Load Data
'''
# print('Loading Data...')
logger.info("Loading Data...")
data = helper.SimulatedDataset(path=config.Path,
                               label_filename=config.LabelFilename,
                               spec_filename=config.SpecFilename,
                               transform=spec_transform,
                               target_transform=label_transform)
################
## Split Data ##
################
length_dataset = len(data)  # get length of data
logger.info(f"Size of dataset: {length_dataset}")

# get ratios
test_size = int(0.1 * length_dataset)                       # amount of test data (10%)
validation_size = int (0.1 * length_dataset)                # amount of validation data (10%) 
train_size = length_dataset - test_size - validation_size   # amount of training and validation data (80%)

logger.info(f"Size of training data:   {train_size}")
logger.info(f"Size of validation data: {validation_size}")
logger.info(f"Size of test data:       {test_size}")

# split 
train_data, validation_data, test_data = random_split(data, [train_size, validation_size, test_size])   # split data

# Data Loaders
train_loader = DataLoader(train_data, batch_size = config.BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size = config.BATCH_SIZE, shuffle=False)
logger.info("Finished loading data!")


'''
Training 
'''
logger.info(f"Starting training...")
########################
## loss and optimizer ##
########################
# criterion = nn.CrossEntropyLoss() 
criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
loss_values = []

# num_total_steps = len(train_loader)
for epoch in range(config.NUM_EPOCHS):     # iterate over epochs
    model.train()       
    for i, (spectrograms, labels) in enumerate(train_loader): # iterate over spectrograms and labels of train_loader
        # make spectrograms float for compatability with the model
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
            logger.info(f"Epoch {epoch+1} / {config.NUM_EPOCHS}, Step {i+1}, Loss = {loss.item():.10f}")
        # Write loss into array
        loss_values.append(loss.item())
    scheduler.step()
    new_lr = scheduler.get_lr()
    logger.info(f"New learning rate: {new_lr}")

# vis.save_plot_training_loss(loss_values, f"{config.loss_plot_filepath}")
# logger.info(f"Saved plot of training loss for {fold+1}!")
logger.info("Training finished!")

model.eval()

logger.info("Starting Validation")
with torch.no_grad():
    val_losses = []
    for spectrograms, labels in validation_loader:
        spectrograms = spectrograms.float().to(device)
        labels = labels.float().to(device)

        outputs = model(spectrograms)
        val_loss = criterion(outputs, labels)
        val_losses.append(val_loss.item())

    avg_val_loss = np.mean(val_losses)
    logger.info(f"Validation Loss: {avg_val_loss:.10f}")

# Write state_dict of model to file
torch.save(model.state_dict(), config.model_filepath)
logger.info("Saved Model")

# '''
# Testing
# '''
logger.info("Starting Test Step...")
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
test_losses = []

model.eval()
with torch.no_grad():
    for spectrograms, labels in test_loader:
        spectrograms = spectrograms.float().to(device)
        labels = labels.float().to(device)

        outputs = model(spectrograms)
        test_loss = criterion(outputs, labels)
        test_losses.append(test_loss.item())

    avg_test_loss = np.mean(test_losses)
    logger.info(f"Test Loss: {avg_test_loss:.10f}")

    if len(test_data) > 0:
        test_sample = random.choice(test_data)
        spectrogram, label = test_sample
        spectrogram = spectrogram.float().unsqueeze(0).to(device)
        label = label.float().cpu().numpy().flatten()
        label = label_unscaler(label)

        prediction = model(spectrogram).cpu().numpy().flatten()
        prediction = label_unscaler(prediction)
        original_label = label.cpu().numpy().flatten()
        vis.compareTimeDomain("./random_test_prediction.png", original_label, prediction)

logger.info("Test Step finished!")

for handler in logger.handlers:
    handler.flush()
    handler.close()
