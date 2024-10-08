'''
training.py Script

Script containing the training of the pulse reconstruction model
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
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering import matplotlib.pyplot as plt

import random
import logging

from sklearn.model_selection import KFold
# Classes, methods and functions from different files
import helper
import loss
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
logger.info(config.DESCRIPTOR)
logger.info(f"Training TBDrms retrieval from spectrogram")
logger.info(f"Writing into log file: {config.log_filepath}")
logger.info(f"Dataset used: {config.Path}")
logger.info(f"Spectrograms used: {config.SpecFilename}")
logger.info(f"Size of output tensor: {2*config.OUTPUT_SIZE} elements")
logger.info(f"Batch size: {config.BATCH_SIZE} elements")
logger.info(f"Number of epochs: {config.NUM_EPOCHS}")
logger.info(f"Initial learning rate: {config.LEARNING_RATE}")
logger.info(f"Only Pulses with PBDrms lower than {config.TBDRMS_THRESHOLD} are used!")

# Transforms (Inputs)
# Resample the spectrograms
spec_read = helper.ReadSpectrogram()
spec_resample = helper.ResampleSpectrogram(
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_FREQUENCIES,
    config.OUTPUT_START_FREQUENCY,
    config.OUTPUT_END_FREQUENCY,
    type='wavelength'
    )
spec_transform = transforms.Compose([spec_read, spec_resample])

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
model = helper.CustomDenseNetTBDrms()
# model.load_state_dict(torch.load('./models/trained_model_3.pth', weights_only=True))
model.float()
model.to(device)
model.eval()
# Freeze the layers before self.densenet
for param in model.densenet.parameters():
    param.requires_grad = False

# Only allow gradients on the layers after densenet
for param in model.fc1.parameters():
    param.requires_grad = True

for param in model.fc2.parameters():
    param.requires_grad = True
logger.info("Freezing early layers!")
logger.info("Loading Model finished!")

'''
Load Data
'''
# print('Loading Data...')
logger.info("Loading Data...")
data = helper.LoadDatasetTBDrms(
        path=config.Path,
        tbd_filename=config.TBDrmsFilename,
        spec_filename=config.SpecFilename,
        transform=spec_transform
        )
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
# criterion = loss.PulseRetrievalLossFunction(
        # penalty_factor=config.PENALTY_FACTOR,
        # threshold=config.PENALTY_THRESHOLD
        # )

# optimizer used
# optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
optimizer = torch.optim.Adam(
        [   
         {'params': model.fc1.parameters()},
         {'params': model.fc2.parameters()}
        ],
        lr=config.LEARNING_RATE,
	    weight_decay=config.WEIGHT_DECAY
	    )
# optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
# optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=0.9)
#optimizer = torch.optim.SGD(
        # [   
        # { 'params': model.fc1.parameters()},
        # {'params': model.fc2.parameters()}
        # ],
        # lr=config.LEARNING_RATE,
	#momentum = 0.9)

# scheduler for changing learning rate after each epoch
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.GAMMA_SCHEDULER)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5,  # Factor by which the learning rate will be reduced (e.g., half the learning rate).
    patience=2,  # Number of epochs with no improvement to wait before reducing the learning rate.
    threshold=0.0001   # Threshold for measuring the new optimum.
    )

# list containing all loss values
training_losses = []
validation_losses = []
learning_rates = []

# save initial learning_rate
learning_rates.append(config.LEARNING_RATE)

for epoch in range(config.NUM_EPOCHS):     # iterate over epochs
    model.train()       
    for i, (spectrograms, labels) in enumerate(train_loader): # iterate over spectrograms and labels of train_loader
            # make spectrograms float for compatability with the model
            # spectrograms = spectrograms.float()
        # send spectrogram and label data to selected device
        spectrograms = spectrograms.float().to(device)  # [tensor]
        labels = labels.float().to(device)      # [tensor]
        
        # Forward pass
        outputs = model(spectrograms)   # [tensor]
        # loss = criterion(outputs, labels)   # [float]
        loss = criterion(outputs.squeeze(), labels.squeeze())   # [float]

        # Backward pass
        loss.backward()
	# Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Print information (every config.TRAINING_LOG_STEP_SIZE steps)
        if (i+1) % config.TRAINING_LOG_STEP_SIZE == 0:
            # print(f'Epoch {epoch+1} / {NUM_EPOCHS}, Step {i+1} / {num_total_steps}, Loss = {loss.item():.10f}')
            logger.info(f"Epoch {epoch+1} / {config.NUM_EPOCHS}, Step {i+1} / {int(train_size/config.BATCH_SIZE)}, Loss = {loss.item():.10e}")
        # Write loss into array
        training_losses.append(loss.item())
    # if (epoch < config.NUM_EPOCHS-1):
        # get new learning_rate
        # When using ExponentialLR
        # scheduler.step()
        # write new learning rate in variable
        # new_lr = scheduler.get_last_lr()[0]
        # save new learning_rate 
        # learning_rates.append(new_lr)
        # logger.info(f"New learning rate: {new_lr}")

        # After the unfreeze_epoch, unfreeze the earlier layers and update the optimizer    
    if (epoch == config.UNFREEZE_EPOCH - 1):
        logger.info("Unfreezing earlier layers")
        
        # Unfreeze all layers
        for param in model.densenet.parameters():
            param.requires_grad = True
        
        # Update optimizer to include all parameters of the model
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,  # Factor by which the learning rate will be reduced (e.g., half the learning rate).
            patience=2,  # Number of epochs with no improvement to wait before reducing the learning rate.
            threshold=0.0001  # Threshold for measuring the new optimum.
            )
        # When using ExponentialLR
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.GAMMA_SCHEDULER)

    logger.info(f"Starting Validation for epoch {epoch+1} / {config.NUM_EPOCHS}")
    model.eval()    # put model into evaluation mode
    with torch.no_grad():   # disable gradient computation for evaluation
        for spectrograms, labels in validation_loader:  # iterate over all spectrograms and labels loaded by the validation loader
            spectrograms = spectrograms.float().to(device)  # send spectrogram to device
            labels = labels.float().to(device)  # send label to device

            outputs = model(spectrograms)   # calculate prediction
            validation_loss = criterion(outputs.squeeze(), labels.squeeze())   # calcultate validation loss
            validation_losses.append(validation_loss.item())  # plave validation loss into list

        avg_val_loss = np.mean(validation_losses)  # calculate validation loss for this epoch
    logger.info(f"Validation Loss: {avg_val_loss:.10e}")
    if epoch < config.NUM_EPOCHS-1:
        # When using ReduceLROnPlateau
        scheduler.step(avg_val_loss)
        # write new learning rate in variable
        new_lr = scheduler.get_last_lr()[0]
        # save new learning_rate 
        learning_rates.append(new_lr) 
        logger.info(f"New learning rate: {new_lr}")

# plot training loss
vis.save_plot_training_loss(
        training_loss = training_losses,
        validation_loss = validation_losses,
        learning_rates = learning_rates,
        train_size = train_size // config.BATCH_SIZE,
        num_epochs = config.NUM_EPOCHS,
        filepath = f"{config.loss_plot_filepath}"
        )
logger.info("Training finished!")

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
        test_loss = criterion(outputs.squeeze(), labels.squeeze())
        test_losses.append(test_loss.item())

    avg_test_loss = np.mean(test_losses)
    logger.info(f"Test Loss: {avg_test_loss:.10e}")

    if len(test_data) > 0:
        # get a random sample
        test_sample = random.choice(test_data)
        spectrogram, label = test_sample
        # adding an extra dimension to spectrogram and label to simulate a batch size of 1
        spectrogram = spectrogram.unsqueeze(0)
        label = label.unsqueeze(0).numpy()
        # send spectrogram to device and make prediction
        spectrogram = spectrogram.float().cpu()
        prediction = model(spectrogram).cpu().numpy()
        logger.info(f"Testing Step: Random Predicted TBDrms: {prediction:.4e}")
        logger.info(f"Testing Step: Random Label TBDrms:     {label:.4e}")
        logger.info(f"Testing Step: Difference TBDrms:       {torch.abs(label-prediction):.4e}")

logger.info("Test Step finished!")

for handler in logger.handlers:
    handler.flush()
    handler.close()
