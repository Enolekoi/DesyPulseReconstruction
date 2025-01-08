'''
config.py Module

Module used for changing training parameters
'''
#############
## Imports ##
#############
import os
import re
import logging

from modules import constants as c
'''
Training Options

Options that configure parameters for the training process
'''
NUM_EPOCHS = 1      # Number of epochs to train the model
OUTPUT_SIZE = 256   # Amount of samples used for the reconstructed pulse [model output size should be OUTPUT_SIZE]
BATCH_SIZE = 10     # Amount of data points trained at each step
UNFREEZE_EPOCH = 0  # Epoch after which the whole model is trained (before that only the output layers are trained)
LEARNING_RATE = 1.78e-3    # Learning rate at the beginning of training
MAX_LEARNING_RATE = 1.78e-3
WEIGHT_DECAY = 1e-5     # TODO find description
GAMMA_SCHEDULER = 0.9   # Learning rate de-/increases by this factor after each epoch, when using exponential LR decrease
TBDRMS_THRESHOLD = 20   # Only data with a TBDrms higher than this threshold is used for training
DESCRIPTOR = f"Training using variation 2 of hyperparameters , new dataset and no FROG-Error - with {NUM_EPOCHS} Epochs"

'''
Loss function options

Options that configure how the loss function is used
'''
PULSE_THRESHOLD = 0.001     # The Pulse is considered to be between the first and last value over the threshold
PENALTY_FACTOR = 2.0      # Values outside the pulse are surpressed, by weighing their error with this factor
WEIGTH_REAL_PART = 10.0      # Weight used for MSE of the real part
WEIGTH_IMAG_PART = 10.0      # Weight used for MSE of the imaginary part
WEIGTH_INTENSITY = 8.0     # Weight used for MSE of the intensity
WEIGTH_PHASE = 0.0          # Weight used for MSE of the phase (only considered, when there is a pulse)
WEIGTH_FROG_ERROR = 0.000   # Weight used for the FROG Error (if it is 0.0, the calculation is skipped)

'''
Scaling Options

Options for scaling of labels, etc.
'''
MAX_VALUE = 1   # The highest possible value the label can be


'''
Resampled Matrix configuration

Options that configure how the resampled SHG-matrixes are created
'''
OUTPUT_NUM_DELAYS = 512                         # Number of delays the SHG-matrix get resampled to
OUTPUT_NUM_WAVELENGTH = 512                     # Number of delays the SHG-matrix get resampled to
OUTPUT_NUM_FREQUENCIES = OUTPUT_NUM_WAVELENGTH  # Number of frequencies the SHG-matrix get resampled to
OUTPUT_TIMESTEP = 11*c.femto                   # Size of timestep between delays [fs]
OUTPUT_START_WAVELENGTH = 478*c.nano            # Smallest wavelength in the dataset [nm]
OUTPUT_END_WAVELENGTH = 561*c.nano              # Largest wavelength in the dataset [nm]
OUTPUT_START_FREQUENCY = c.c2pi / OUTPUT_START_WAVELENGTH    # convert start wavelength to frequency [Hz]
OUTPUT_END_FREQUENCY = c.c2pi / OUTPUT_END_WAVELENGTH        # convert stop wavelength to frequency [Hz]

logger = logging.getLogger(__name__)
'''
getFilepaths()

Description:
    Search for the highest index of a matching subdirectory and create new directory. Return its path
Input:
    root_directory      -> [string] directory containing the subdirectories
    subdirectory_string -> [string] name of subdirectory without numerical content
Output:
    new_directory_path  -> [string] path of newly created subdirectory
'''
def getFilepaths(root_directory, subdirectory_string):
    # Check if root directory exists
    if not os.path.exists(root_directory):
        raise FileNotFoundError(f"The specified root directory doesn't exist: {root_directory}")

    # Regex to match subdirectories that start with the 'subdirectory_string' followed by a number
    pattern = re.compile(f"^{re.escape(subdirectory_string)}(\\d+)$")

    # initialize indices of subdirectories that match the pattern
    indices = []
    
    # find all matching subdirectories and extract their index
    for entry in os.listdir(root_directory):
        # get the full path of the entry
        full_path = os.path.join(root_directory, entry)
        # check if entry is a directory
        if os.path.isdir(full_path):
            # use the pattern to check if the entry fits it
            match = pattern.match(entry)
            # get index if the pattern matches
            if match:
                indices.append(int(match.group(1)))

    # Determine the next highest index
    next_index = max(indices, default=0) + 1

    # Create the new directory
    new_directory_name = f"{subdirectory_string}{next_index}"
    new_directory_path = os.path.join(root_directory, new_directory_name)
    os.makedirs(new_directory_path)

    return new_directory_path

'''
Path Options

The paths used are defined here
'''
ModelPath = "./logs/training_002/model.pth"  # path of pretrained model used to initialize weights before training

LogDirectory = "./logs/"
SubDirectoryString = "log_"
LogSubdiretory = getFilepaths(root_directory = LogDirectory, subdirectory_string=SubDirectoryString)
LearningRateFilePath        = os.path.join(LogSubdiretory,"learning_rate.csv")
LossPlotFilePath            = f"{LogSubdiretory}/loss"
LogFilePath                 = f"{LogSubdiretory}/training.log"
ModelFilePath               = f"{LogSubdiretory}/model.pth"
TrainingLossFilePath        = f"{LogSubdiretory}/training_loss.csv"
ValidationLossFilePath      = f"{LogSubdiretory}/validation_loss.csv"
TestLossFilePath            = f"{LogSubdiretory}/test_loss.csv"
RandomPredictionFilePath    = f"{LogSubdiretory}/random_prediction"
MinPredictionFilePath       = f"{LogSubdiretory}/prediction_min"
MaxPredictionFilePath       = f"{LogSubdiretory}/prediction_max"
MeanPredictionFilePath      = f"{LogSubdiretory}/prediction_mean"
LearningRateFinderFilePath  = f"{LogSubdiretory}/lr_finder_plot"

Path = "/mnt/data/desy/frog_simulated/grid_256_v4/" # Path to data used for training 
# Path = "/mnt/data/desy/dataset/training_data/"
TBDrmsFilename = "./grid_256_v4_TBD.csv"     # Path to a sorted list of all directories and their corresponfing TBDrms
ShgFilename = "as_gn00.dat"    # Filename of the file containing the SHG-matrix
LabelFilename = "Es.dat"        # Filename of the file containing the label
