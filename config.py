'''
config.py Module

Module used for changing training parameters
'''
import logging
import numpy as np
import os
import torch

DESCRIPTOR = "Testing training grid_256_v3 - regular error as loss function , with 100 Epochs"
OUTPUT_SIZE = 256   # Amount of samples used for the reconstructed pulse [model output size should be 2*OUTPUT_SIZE]
BATCH_SIZE = 10     # Amount of spectrograms trained at each step
NUM_EPOCHS = 100     # Number of epochs to train the model
UNFREEZE_EPOCH = 70 # Epoch after which the whole model is trained (before that only the output layers are trained)
LEARNING_RATE = 1e-3    # Learning rate at the beginning of training
WEIGHT_DECAY = 1e-5
GAMMA_SCHEDULER = 0.9   # Learning rate de-/increases by this factor after each epoch
TRAINING_LOG_STEP_SIZE = BATCH_SIZE
TBDRMS_THRESHOLD = 0.3  # Only data with a TBDrms higher than this threshold is used for training

PENALTY_FACTOR = 1
PENALTY_THRESHOLD = 1
MAX_REAL = 1    # The highest possible value the real part of the signal can be
MAX_IMAG = 1    # The highest possible value the imaginary part of the signal can be

OUTPUT_NUM_DELAYS = 512         # Number of delays the spectrograms get resampled to
OUTPUT_NUM_FREQUENCIES = 512    # Number of frequency points the spectrograms get resampled to
OUTPUT_TIMESTEP = 1.5e-15           # Size of timestep between delays [fs]
OUTPUT_START_WAVELENGTH = 226e-9  # Smallest wavelength in the dataset [nm]
# OUTPUT_START_WAVELENGTH = 504   # [nm]
# OUTPUT_END_WAVELENGTH = 451     # Highest wavelength in the dataset [nm]
OUTPUT_END_WAVELENGTH = 528e-9     # [nm]
OUTPUT_START_FREQUENCY = (299792458  * 2 * torch.pi) / OUTPUT_START_WAVELENGTH    # convert start wavelength to frequency [Hz]
OUTPUT_END_FREQUENCY = (299792458 * 2 * torch.pi) / OUTPUT_END_WAVELENGTH        # convert stop wavelength to frequency [Hz]

ModelPath = "./models/trained_model_7.pth"  # path of pretrained model used to initialize weights before training

ModelDirectory = "./models/"    # directory in which trained models get stored
LogDirectory = "./logs/"        # directory in which training logs get stored
PlotDirectory = "./plots/"        # directory in which training logs get stored
ModelName = "trained_model_"    # base name of trained models
LogName = "training_"           # base name of logs 
TrainingLossPlotName = "training_loss_" # base name of training loss plots
PredicitonPlotName = "post_training_prediction_"    # base name of prediction plot

# Path = "/mnt/data/desy/frog_simulated/grid_512_v2/"
# Path = "/mnt/data/desy/frog_simulated/grid_256/"
Path = "/mnt/data/desy/frog_simulated/grid_256_v3/" # Path to data used for training 
TBDrmsFilename = "./TBDrms_grid_256_v3.csv"     # Path to a sorted list of all directories and their corresponfing TBDrms
SpecFilename = "as_gn00.dat"    # Filename of the file containing the spectrograms
# SpecFilename = "as.dat"
LabelFilename = "Es.dat"        # Filename of the file containing the labels


logger = logging.getLogger(__name__)

'''
getFilepathIndex()
Search the model and log directories for the highest index and determine the next filenames
Input:
    model_directory         -> directory that contains the trained models
    log_directory           -> directory that contains the log files and loss plots
    plot_direcory           -> directory that contains the plots of predicted pulses
    model_base_filename     -> filename of saved models without index and file ending
    log_base_filename       -> filename of logs without index and file ending
    loss_plot_base_filename -> filename of loss plots without index and file ending
    prediction_base_filename-> filename of prediction plot without index and file ending
Output:
    model_filepath      -> filepath of the next model file to be saved
    log_filepath        -> filepath of the next log file to be saved
    loss_plot_filepath  -> filepath of the next loss plot file to be saved
    prediction_filepath -> filepath of the next prediction plot file to be saved
'''
def getFilepathIndex(model_directory, log_directory, plot_directory, model_base_filename, log_base_filename, loss_plot_base_filename, prediction_base_filename):

    # Check if log, model and plot directory exists, if not create them
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    
    # Get a list of all files in the directory
    model_files = os.listdir(model_directory)
    log_files = os.listdir(log_directory)
    plot_files = os.listdir(plot_directory)

    # Filter out matching file names and write them in a list
    matching_model_files = [f for f in model_files if f.startswith(model_base_filename) and f.endswith('.pth')]
    matching_loss_plot_files = [f for f in log_files if f.startswith(loss_plot_base_filename) and f.endswith('.png')]
    matching_log_files= [f for f in log_files if f.startswith(log_base_filename) and f.endswith('.log')]
    matching_prediction_files = [f for f in plot_files if f.startswith(prediction_base_filename) and f.endswith('.png')]

    # Determin what the next model index is
    if matching_model_files:
        model_numbers = [int(f[len(model_base_filename):-4]) for f in matching_model_files if f[len(model_base_filename):-4].isdigit()]
        next_model_index = max(model_numbers) + 1 if model_numbers else 1
    else:
        next_model_index = 1

    # Determin what the log index is
    if matching_log_files:
        log_numbers = [int(f[len(log_base_filename):-4]) for f in matching_log_files if f[len(log_base_filename):-4].isdigit()]
        next_log_index = max(log_numbers) + 1 if log_numbers else 1
    else:
        next_log_index = 1

    # Determin what the next loss plot index is
    if matching_loss_plot_files:
        loss_plot_numbers = [int(f[len(loss_plot_base_filename):-4]) for f in matching_loss_plot_files if f[len(loss_plot_base_filename):-4].isdigit()]
        next_loss_plot_index = max(loss_plot_numbers) + 1 if loss_plot_numbers else 1
    else:
        next_loss_plot_index = 1

    # Determin what the next prediction plot index is
    if matching_prediction_files:
        prediction_plot_numbers = [int(f[len(prediction_base_filename):-4]) for f in matching_prediction_files if f[len(prediction_base_filename):-4].isdigit()]
        next_prediction_plot_index = max(prediction_plot_numbers) + 1 if prediction_plot_numbers else 1
    else:
        next_prediction_plot_index = 1

    # get the largest index
    next_index = max(next_model_index, next_loss_plot_index, next_log_index, next_prediction_plot_index)

    # Get the new filename
    new_model_filename = f"{model_base_filename}{next_index}.pth"
    new_loss_plot_filename = f"{loss_plot_base_filename}{next_index}.png"
    new_log_filename = f"{log_base_filename}{next_index}.log"
    new_prediction_filename = f"{prediction_base_filename}{next_index}.png"

    # Join path and filenames together
    model_filepath = os.path.join(model_directory, new_model_filename)
    loss_plot_filepath = os.path.join(log_directory, new_loss_plot_filename)
    log_filepath = os.path.join(log_directory, new_log_filename)
    prediction_filepath = os.path.join(plot_directory, new_prediction_filename)

    return model_filepath, log_filepath, loss_plot_filepath, prediction_filepath

####################
## get file paths ##
####################
model_filepath, log_filepath, loss_plot_filepath, prediction_filepath = getFilepathIndex(
        model_directory=ModelDirectory,
        log_directory=LogDirectory,
        plot_directory=PlotDirectory,
        model_base_filename=ModelName,
        log_base_filename=LogName,
        loss_plot_base_filename=TrainingLossPlotName,
        prediction_base_filename=PredicitonPlotName
)
