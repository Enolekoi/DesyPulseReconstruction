'''
training.py Script

Script containing the training of the pulse reconstruction model
'''
#############
## Imports ##
#############
import random
import logging
import matplotlib 
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from modules import helper
from modules import loss as loss_module
from modules import visualize as vis
from modules import config
from modules import data
from modules import models

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
            logging.StreamHandler()
            ]
)
logger = logging.getLogger(__name__)

# Log some information
logger.info(config.DESCRIPTOR)
logger.info("Supervised Training")
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
model.load_state_dict(torch.load('./logs/log_20/model.pth', weights_only=True, map_location=torch.device('cpu')))

# set the model to float, send it to the selected device and put it in evaluation mode
model.float().to(device).eval()

logger.info("Freezing early layers!")
logger.info("Loading Model finished!")

'''
Load Data
'''
# print('Loading Data...')
logger.info("Loading Data...")
# configure the data loader
data_loader = data.LoadDatasetReconstruction(
        path='../samples/swamp/data/',
        label_filename=config.LabelFilename,
        shg_filename=config.ShgFilename,
        tbdrms_file='./test_TBD.csv',  # Path to the file containing TBDrms values
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
# test_size = int(1000)
# train_size = int(49006)
# train_size = int(0.1 * length_dataset)                       # amount of test data (10%)
validation_size = int (0.1 * length_dataset)                # amount of validation data (10%) 
# validation_size = 0
# validation_size = length_dataset - test_size - train_size   # amount of training and validation data (80%)
train_size = length_dataset - test_size - validation_size   # amount of training and validation data (80%)
# rest_size = length_dataset - test_size - train_size   # amount of training and validation data (80%)
logger.info(f"Size of training data:   {train_size}")
logger.info(f"Size of validation data: {validation_size}")
logger.info(f"Size of test data:       {test_size}")

# split the dataset accordingly
# train_data, rest_data, validation_data, test_data = random_split(data_loader, [train_size, rest_size, validation_size, test_size])   # split data
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

'''
Training 
'''
logger.info("Starting training...")
########################
## loss and optimizer ##
########################
# loss function
# define and configure the loss function
criterion = loss_module.PulseRetrievalLossFunction(
        use_label=config.USE_LABEL, 
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
'''
Testing Loop
'''
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

# don't calculate gradients
with torch.no_grad():
    # get the prediction of a random test data point and plot it
    if len(test_data) > 0:
        # get a random sample
        test_sample = random.choice(test_data)
        shg_matrix, label, header = test_sample
        # adding an extra dimension to shg_matrix and label to simulate a batch size of 1
        shg_matrix = shg_matrix.unsqueeze(0)
        label = label.unsqueeze(0)
        # send shg_matrix to device and make prediction
        shg_matrix = shg_matrix.float().to(device)
        prediction = model(shg_matrix) 
        # send label and prediction to cpu, so that it can be plotted
        label = label_unscaler(label).cpu()
        prediction = label_unscaler(prediction).cpu()
        # calculate the imaginary part of the signal and make it the shape of the label
        prediction_analytical = loss_module.hilbert(prediction.squeeze())
        prediction = torch.cat((prediction_analytical.real, prediction_analytical.imag))
        prediction = label_remove_ambiguieties(prediction)
        # plot
        vis.compareTimeDomainComplex('random_prediction.png', label, prediction)# save learning rate and losses to csv files

logger.info("Test Step finished!")

for handler in logger.handlers:
    handler.flush()
    handler.close()
