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
# label_reader = helper.ReadLabelFromEs(config.OUTPUT_SIZE)
label_reader = helper.ReadPhaseFromEs(config.OUTPUT_SIZE)
# label_scaler = helper.ScaleLabel(max_intensity=config.MAX_INTENSITY, max_phase=config.MAX_PHASE)
label_scaler = helper.Scaler(config.MAX_INTENSITY, config.MAX_PHASE)
# label_transform = transforms.Compose([label_reader, label_phase_correction, label_scaler])
label_transform = transforms.Compose([label_reader, label_scaler.scalePhase])

label_unscaler = label_scaler.scalePhase

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
modelPhase = helper.CustomDenseNet(
    num_outputs=2*config.OUTPUT_SIZE
    )

modelPhase.float()
modelPhase.to(device)
modelPhase.eval()

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
# loss function
criterion = nn.MSELoss()
# optimizer used
optimizer = torch.optim.Adam(modelPhase.parameters(), lr=config.LEARNING_RATE)
# scheduler for changing learning rate after each epoch
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
# list containing all loss values
loss_values = []

for epoch in range(config.NUM_EPOCHS):     # iterate over epochs
    modelPhase.train()       
    for i, (spectrograms, labels) in enumerate(train_loader): # iterate over spectrograms and labels of train_loader
            # make spectrograms float for compatability with the model
            # spectrograms = spectrograms.float()
        # send spectrogram and label data to selected device
        spectrograms = spectrograms.to(device)  # [tensor]
        labels = labels.float().to(device)      # [tensor]
        
        # Forward pass
        outputs = modelPhase(spectrograms)   # [tensor]
        loss = criterion(outputs, labels)   # [float]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print information (every config.TRAINING_LOG_STEP_SIZE steps)
        if (i+1) % config.TRAINING_LOG_STEP_SIZE == 0:
            # print(f'Epoch {epoch+1} / {NUM_EPOCHS}, Step {i+1} / {num_total_steps}, Loss = {loss.item():.10f}')
            logger.info(f"Epoch {epoch+1} / {config.NUM_EPOCHS}, Step {i+1}, Loss = {loss.item():.10f}")
        # Write loss into array
        loss_values.append(loss.item())
    scheduler.step()
    new_lr = scheduler.get_last_lr()
    logger.info(f"New learning rate: {new_lr}")

    logger.info(f"Starting Validation for epoch {epoch+1} / {config.NUM_EPOCHS}")
    modelPhase.eval()    # put model into evaluation mode
    with torch.no_grad():   # disable gradient computation for evaluation
        val_losses = []     # list containing all validation losses (resets after each epoch)
        for spectrograms, labels in validation_loader:  # iterate over all spectrograms and labels loaded by the validation loader
            spectrograms = spectrograms.float().to(device)  # send spectrogram to device
            labels = labels.float().to(device)  # send label to device

            outputs = modelPhase(spectrograms)   # calculate prediction
            val_loss = criterion(outputs, labels)   # calcultate validation loss
            val_losses.append(val_loss.item())  # plave validation loss into list

        avg_val_loss = np.mean(val_losses)  # calculate validation loss for this epoch
    logger.info(f"Validation Loss: {avg_val_loss:.10f}")

# plot training loss
vis.save_plot_training_loss(loss_values, f"{config.loss_plot_filepath}")
logger.info(f"Saved plot of training loss! to {config.loss_plot_filepath}!")
logger.info("Training finished!")

# Write state_dict of model to file
torch.save(modelPhase.state_dict(), config.model_filepath)
logger.info("Saved Model")

# '''
# Testing
# '''
logger.info("Starting Test Step...")
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
test_losses = []

modelPhase.eval()
with torch.no_grad():
    for spectrograms, labels in test_loader:
        spectrograms = spectrograms.float().to(device)
        labels = labels.float().to(device)

        outputs = modelPhase(spectrograms)
        test_loss = criterion(outputs, labels)
        test_losses.append(test_loss.item())

    avg_test_loss = np.mean(test_losses)
    logger.info(f"Test Loss: {avg_test_loss:.10f}")

    if len(test_data) > 0:
        # get a random sample
        test_sample = random.choice(test_data)
        spectrogram, label = test_sample
        # adding an extra dimension to spectrogram and label to simulate a batch size of 1
        spectrogram = spectrogram.unsqueeze(0)
        label = label.unsqueeze(0)
        # send spectrogram to device and make prediction
        spectrogram = spectrogram.float().to(device)
        prediction = modelPhase(spectrogram) 
        # send label and prediction to cpu, so that it can be plotted
        label = label_unscaler(label).cpu()
        prediction = label_unscaler(prediction).cpu()
        vis.comparePhase("./random_test_phase_prediction.png", label, prediction)
        # vis.compareTimeDomainComplex("./random_test_prediction.png", label, prediction)

logger.info("Test Step finished!")

for handler in logger.handlers:
    handler.flush()
    handler.close()
