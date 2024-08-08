'''
Libraries and classes/methods
'''
# Libraries used in this file
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
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
# Paths
# LogDirectory = "./logs/"
# LogName = "training_"
# TrainingLossPlotName = "training_loss_"

Path = "/mnt/data/desy/frog_simulated/grid_256/"
SpecFilename = "as.dat"
LabelFilename = "Es.dat"

# Constants
# get the correct filepaths of all files
log_filepath, loss_plot_filepath = config.getLogFilepath(
        directory=config.LogDirectory,
        log_base_filename=config.LogName,
        loss_plot_base_filename=config.TrainingLossPlotName
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
logger.info(f"Size of Output Tensor: {2*config.OUTPUT_SIZE} Elements")
logger.info(f"Batch Size: {config.BATCH_SIZE} Elements")
logger.info(f"Number of Epochs: {config.NUM_EPOCHS}")
logger.info(f"Learning Rate: {config.LEARNING_RATE}")

# Transforms
spec_transform = helper.ResampleSpectrogram(config.OUTPUT_NUM_DELAYS, config.OUTPUT_TIMESTEP, config.OUTPUT_NUM_WAVELENGTH, config.OUTPUT_START_WAVELENGTH, config.OUTPUT_END_WAVELENGTH)
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
model = helper.CustomDenseNet(output_scale=40, num_outputs=2*config.OUTPUT_SIZE)
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
test_size = int(0.1 * length_dataset)     # amount of validation data (10%)
train_validation_size = length_dataset - test_size # amount of training and validation data (90%)

logger.info(f"Size of training and validation data: {train_validation_size}")
logger.info(f"Size of test data: {test_size}")

# split 
train_validation_data, test_data = random_split(data, [train_validation_size, test_size])   # split data

k_folds = KFold(n_splits=config.NUMBER_FOLDS, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, validation_idx) in enumerate(k_folds.split(train_validation_data)):
    logger.info(f"Starting fold {fold+1}")

    train_subset = Subset(train_validation_data, train_idx)
    validation_subset = Subset(train_validation_data, validation_idx)
    logger.info(f"Starting to load data for fold {fold+1}...")
    # Data Loaders
    train_loader = DataLoader(train_subset, batch_size = config.BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size = config.BATCH_SIZE, shuffle=False)
    logger.info(f"Finished loading data for fold {fold+1}!")

    '''
    Training 
    '''
    logger.info(f"Starting training for fold {fold+1}...")
    ########################
    ## loss and optimizer ##
    ########################
    # criterion = nn.CrossEntropyLoss() 
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
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
                logger.info(f"Fold {fold+1} / {config.NUMBER_FOLDS}, Epoch {epoch+1} / {config.NUM_EPOCHS}, Step {i+1}, Loss = {loss.item():.10f}")
            # Write loss into array
            loss_values.append(loss.item())
    # vis.save_plot_training_loss(loss_values, f"{config.loss_plot_filepath}")
    logger.info(f"Saved plot of training loss for {fold+1}!")
    logger.info(f"Fold {fold+1} Training finished!")
    
    model.eval()
    with torch.no_grad():
        val_losses = []
        for spectrograms, labels in validation_loader:
            spectrograms = spectrograms.float().to(device)
            labels = labels.float().to(device)

            outputs = model(spectrograms)
            val_loss = criterion(outputs, labels)
            val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        fold_results.append(avg_val_loss)
        logger.info(f"Fold {fold+1}, Validation Loss: {avg_val_loss:.10f}")

logger.info(f"Cross-validation finished! Results: {fold_results}")
logger.info(f"Average Validation Loss: {np.mean(fold_results):.10f}")

'''
Testing
'''
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
        label = label.float().to(device)

        prediction = model(spectrogram).cpu().numpy().flatten()
        original_label = label.cpu().numpy().flatten()
        vis.compareTimeDomain("./random_test_prediction.png", original_label, prediction)

logger.info("Test Step finished!")

for handler in logger.handlers:
    handler.flush()
    handler.close()
