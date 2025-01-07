from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from modules import config
from modules import data
from modules import models
from modules import loss as loss_module

'''
find_lr()

Description:
    Find the optimal learning rate
Inputs:
    model           -> [pytorch model] model which to use
    train_loader    -> [pytorch dataloader] model which to use
    criterion       -> [loss function] loss function used in training script
    optimizer       -> [pytorch optimizer] optimizer used in training script
    device          -> [device] cuda/cpu
    init_lr         -> [float] lowest learning rate to test
    final_lr        -> [float] highest learning rate to test
    beta            -> [float] 
'''
def find_lr(
    model, train_loader, criterion, optimizer, device, init_lr=1e-8, final_lr=10.0, beta=0.98
):
    logger.info("Starting learning rate finder")
    # get the number of iterations in the training dataset 
    num = len(train_loader) - 1
    # multiplicative factor for increasing the learning rate
    multiplicative_factor = (final_lr / init_lr) ** (1 / num)
    # initialize learning rate
    lr = init_lr
    logger.info(f"Learning Rate = {lr}")
    # set initial learning in the optimizer
    optimizer.param_groups[0]['lr'] = lr
    # initialize smoothed and best losses
    avg_loss = 0.0
    best_loss = float('inf')
    # list to store learning rate and losses
    losses, lrs = [], []
    model.to(device)

    # iterate over the training data
    for i, (shg_matrix, label, header) in enumerate(train_loader):
        logger.info(f"Step {i}/{num}")
        logger.info(f"Learning Rate = {lr}")
        # move the data to the selected device
        shg_matrix, label, header = shg_matrix.to(device), label.to(device), header.to(device)
        # zero out gradients
        optimizer.zero_grad()
        # forward pass through the model
        outputs = model(shg_matrix)
        # compute the loss 
        loss = criterion(
            prediction=outputs.to(device), 
            label=label.to(device), 
            shg_matrix=shg_matrix.to(device), 
            header=header.to(device)
            )

        # smooth out the loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** (i + 1))
        logger.info(f"Average Loss  = {avg_loss}")
        logger.info(f"Smoothed Loss = {smoothed_loss}")

        # Stop if the loss increases too much (indicating instability)
        if i > 1 and smoothed_loss > 4 * best_loss:
            break

        # Update the best loss encountered so far
        if smoothed_loss < best_loss or i == 1:
            best_loss = smoothed_loss
            logger.info(f"New best loss: {best_loss} with learning rate: {lr}")

        # Store the smoothed loss and current learning rate
        losses.append(smoothed_loss)
        lrs.append(lr)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        # increase the used learning rate
        lr *= multiplicative_factor
        optimizer.param_groups[0]['lr'] = lr

    return lrs, losses

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
            logging.FileHandler(config.LogFilePath),
            logging.StreamHandler()
            ]
)
logger = logging.getLogger(__name__)

# Log some information
logger.info("Learning rate finder")
logger.info(f"Writing into log file: {config.LogFilePath}")
logger.info(f"Dataset used: {config.Path}")
logger.info(f"SHG-matrix used: {config.ShgFilename}")
logger.info(f"Size of output tensor: {2*config.OUTPUT_SIZE} elements")
logger.info(f"Batch size: {config.BATCH_SIZE} elements")
logger.info(f"Number of epochs: {config.NUM_EPOCHS}")
logger.info(f"Initial learning rate: {config.LEARNING_RATE}")
logger.info(f"Only Pulses with PBDrms lower than {config.TBDRMS_THRESHOLD} are used!")

# Transforms (Inputs)
# Read the SHG-matrix and their headers
shg_read = data.ReadSHGmatrix()
shg_restructure = data.CreateAxisAndRestructure()
# Resample the SHG-matrix to the same delay and wavelength axes
shg_resample = data.ResampleSHGmatrix(
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_WAVELENGTH,
    config.OUTPUT_START_WAVELENGTH,
    config.OUTPUT_END_WAVELENGTH,
    )
shg_3channel = data.Create3ChannelSHGmatrix()
shg_transform = transforms.Compose([shg_read, shg_resample, shg_3channel])

# Transforms (Labels)
# Read the Labels
label_reader = data.ReadLabelFromEs(config.OUTPUT_SIZE)
# Remove the trivial ambiguities from the labels
label_remove_ambiguieties = data.RemoveAmbiguitiesFromLabel(config.OUTPUT_SIZE)
# Scale the Labels to the correct amplitude
scaler = data.Scaler(
    number_elements=config.OUTPUT_SIZE, 
    max_value=config.MAX_VALUE
    )
# scale to [0, 1] 
label_scaler = scaler.scale
# scale to original label size
label_unscaler = scaler.unscale
label_transform = transforms.Compose([label_reader, label_remove_ambiguieties, label_scaler])

# If cuda is is available use it instead of the cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device used (cuda/cpu): {device}")
if device == 'cuda':
    torch.cuda.empty_cache()

'''
Load Model
'''
logger.info("Loading Model...")
# Load custom DenseNet
model = models.CustomDenseNetReconstruction(
    num_outputs=config.OUTPUT_SIZE
    )
# Define location of pretrained weights
# model.load_state_dict(torch.load('./models/trained_model_3.pth', weights_only=True))

# set the model to float, send it to the selected device and put it in evaluation mode
model.float().to(device).eval()

# Freeze the layers of the densenet
for param in model.densenet.parameters():
    param.requires_grad = False

# Only allow gradients on the last layers after the densenet
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
# configure the data loader
data_loader = data.LoadDatasetReconstruction(
        path=config.Path,
        label_filename=config.LabelFilename,
        shg_filename=config.ShgFilename,
        tbdrms_file=config.TBDrmsFilename,  # Path to the file containing TBDrms values
        tbdrms_threshold=config.TBDRMS_THRESHOLD,  # TBDrms threshold for filtering    
        transform=shg_transform,
        target_transform=label_transform
        )
################
## Split Data ##
################
# get the length of the dataset
length_dataset = len(data_loader)
logger.info(f"Size of dataset: {length_dataset}")

# get ratios of train, validation and test data
test_size = int(0.1 * length_dataset)                       # amount of test data (10%)
validation_size = int (0.1 * length_dataset)                # amount of validation data (10%) 
train_size = length_dataset - test_size - validation_size   # amount of training and validation data (80%)
logger.info(f"Size of training data:   {train_size}")

# split the dataset accordingly
train_data, validation_data, test_data = random_split(data_loader, [train_size, validation_size, test_size])   # split data

# define the data loaders for training and validation
train_loader = DataLoader(train_data, batch_size = config.BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size = config.BATCH_SIZE, shuffle=False)
logger.info("Finished loading data!")

# get the number of steps per epoch
if train_size % config.BATCH_SIZE  != 0:
    NUM_STEPS_PER_EPOCH = (train_size // config.BATCH_SIZE + 1)
else:
    NUM_STEPS_PER_EPOCH = (train_size // config.BATCH_SIZE)
# get the number of total steps 
NUM_STEPS = NUM_STEPS_PER_EPOCH * config.NUM_EPOCHS

########################
## loss and optimizer ##
########################
# loss function
# define and configure the loss function
# criterion = nn.MSELoss()
criterion = loss_module.PulseRetrievalLossFunction(
        pulse_threshold = config.PULSE_THRESHOLD,
        penalty = config.PENALTY_FACTOR,
        real_weight = config.WEIGTH_REAL_PART,
        imag_weight = config.WEIGTH_IMAG_PART,
        intensity_weight = config.WEIGTH_INTENSITY,
        phase_weight = config.WEIGTH_PHASE,
        frog_error_weight= config.WEIGTH_FROG_ERROR
        )
logger.info(f"Threshold over which signal is considered part of the pulse: {config.PULSE_THRESHOLD}")
logger.info(f"Penalty for signal outside of pulse:  {config.PENALTY_FACTOR}")
logger.info(f"Weight Used for MSE of Real part:     {config.WEIGTH_REAL_PART}")
logger.info(f"Weight Used for MSE of imaginary part:{config.WEIGTH_IMAG_PART}")
logger.info(f"Weight Used for MSE of Intensity:     {config.WEIGTH_INTENSITY}")
logger.info(f"Weight Used for MSE of Phase:         {config.WEIGTH_PHASE}")
logger.info(f"Weight Used for FROG-Error:           {config.WEIGTH_FROG_ERROR}")

# define and configure the optimizer used
optimizer = torch.optim.AdamW(
        [   
         {'params': model.fc1.parameters()},
         {'params': model.fc2.parameters()}
        ],
        lr=config.LEARNING_RATE,
	    weight_decay=config.WEIGHT_DECAY
	    )

# Run the learning rate finder
lrs, losses = find_lr(
    model=model,
    train_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    init_lr=1e-8,  # Small initial learning rate
    final_lr=10.0  # Large final learning rate
)

# get correct paths for the plot
png_filepath = os.path.join(config.LearningRateFinderFilePath, ".png")

# Plot the learning rate vs. loss curve
plt.figure(figsize=(10, 6))  # Set figure size
plt.plot(lrs, losses)  # Plot learning rates (x-axis) against losses (y-axis)
plt.xscale('log')  # Use logarithmic scale for learning rates
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.grid(True)
plt.savefig(png_filepath)  # Save the plot to a file
logger.info(f"Learning Rate Finder complete. Saved plot at '{png_filepath}'")
plt.show()  # Display the plot

# Choose the optimal learning rate (manually or programmatically)
optimal_lr = lrs[np.argmin(losses) // 2]  # Example heuristic: take half the learning rate at minimum loss
logger.info(f"Selected optimal learning rate: {optimal_lr}")
