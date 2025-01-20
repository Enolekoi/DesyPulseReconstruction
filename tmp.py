'''
tmp.py Script

Script containing the training of the pulse reconstruction model
'''
#############
## Imports ##
#############
import pandas as pd
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
# CSV-Dateien laden 37
output_csv_path = 'test_results_1.csv'  # Der gewünschte Speicherpfad
model_path = './logs/log_1/model.pth'

test_tbd_path = 'test_TBD.csv'        # Pfad zur test_tbd.csv


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
model.load_state_dict(torch.load(model_path , weights_only=True, map_location=torch.device('cpu')))

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
        path="../samples/swamp/data/",
        label_filename=config.LabelFilename,
        shg_filename=config.ShgFilename,
        tbdrms_file="./test_TBD.csv",  # Path to the file containing TBDrms values
        tbdrms_threshold=20,  # TBDrms threshold for filtering    
        transform=shg_transform,
        target_transform=label_transform
        )
################
## Split Data ##
################
# get the length of the dataset
length_dataset = len(data_loader)
logger.info(f"Size of dataset: {length_dataset}")

test_size = int(100)
rest_size = length_dataset - test_size
logger.info(f"Size of test data:       {test_size}")

# split the dataset accordingly
test_data, rest_data = random_split(data_loader, [test_size, rest_size])   # split data

criterion = loss_module.PulseRetrievalLossFunction(
        use_label=True, 
        pulse_threshold =1e-20,
        penalty = 1.0,
        real_weight = 1.0, 
        imag_weight = 0,
        intensity_weight = 0,
        phase_weight = 0,
        frog_error_weight= 0
        )
'''
Testing Loop
'''
print(data_loader.data_dirs)
logger.info("Starting Test Step...")
test_loader = DataLoader(data_loader, batch_size=1, shuffle=False)
test_losses = []
test_dirs = []

max_index = 99
model.eval()
# don't calculate gradients
with torch.no_grad():
    # iterate over test data
    for idx, (shg_matrix, label, header) in enumerate(test_loader):
        print(f"Index: {idx+1} / {max_index+1}")
        if idx > max_index:
            logger.info(f"Reached the specified index: {max_index}. Stopping iteration.")
            break
        # convert shg_matrix and labels to float and send them to the device
        subdir = data_loader.data_dirs[idx]
        shg_matrix = shg_matrix.float().to(device)
        label = label.float().to(device)
        
        # calculate the predicted output
        outputs = model(shg_matrix)
        # get the loss
        test_loss = criterion(
                prediction=outputs,
                label=label, 
                shg_matrix=shg_matrix, 
                header=header
                )
        # place the loss in a list
        test_losses.append(test_loss.item())
        test_dirs.append(subdir)

    # calculate the mean test loss
    avg_test_loss = np.mean(test_losses)
    logger.info(f"Test Loss: {avg_test_loss:.10e}")

# Combine the test file paths and losses into a DataFrame
test_results = pd.DataFrame({
    'File_Path': test_dirs,
    'Test_Error': test_losses
})

# Save the results to a CSV file
test_results.to_csv(output_csv_path, index=False)

# DataFrames einlesen
test_result = pd.read_csv(output_csv_path)
test_tbd = pd.read_csv(test_tbd_path)

# Sicherstellen, dass die relevanten Spalten vorhanden sind
if 'File_Path' not in test_result.columns:
    raise ValueError("Die Spalte 'File_Path' fehlt in test_result.csv.")
if test_tbd.shape[1] < 4:
    raise ValueError("test_tbd.csv hat weniger als 4 Spalten.")

# Dictionary aus test_tbd.csv erstellen (erste Spalte als Schlüssel, vierte Spalte als Wert)
tbd_mapping = dict(zip(test_tbd.iloc[:, 0], test_tbd.iloc[:, 3]))

# TBD-Werte basierend auf dem Mapping hinzufügen
test_result['TBD'] = test_result['File_Path'].map(tbd_mapping)

# Ergebnis abspeichern
output_path = output_csv_path
test_result.to_csv(output_path, index=False)
logger.info(f"Test results saved to {output_csv_path}")
logger.info("Test Step finished!")

for handler in logger.handlers:
    handler.flush()
    handler.close()
