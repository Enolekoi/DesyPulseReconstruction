'''
config.py Module

Module used for changing training parameters
'''
#############
## Imports ##
#############
import os
import logging

from modules import constants as c
'''
Training Options

Options that configure parameters for the training process
'''
NUM_EPOCHS = 1      # Number of epochs to train the model
OUTPUT_SIZE = 512   # Amount of samples used for the reconstructed pulse [model output size should be OUTPUT_SIZE]
BATCH_SIZE = 10     # Amount of data points trained at each step
UNFREEZE_EPOCH = 30 # Epoch after which the whole model is trained (before that only the output layers are trained)
LEARNING_RATE = 1e-6    # Learning rate at the beginning of training
MAX_LEARNING_RATE = 1e6
WEIGHT_DECAY = 1e-5     # TODO find description
GAMMA_SCHEDULER = 0.9   # Learning rate de-/increases by this factor after each epoch, when using exponential LR decrease
TBDRMS_THRESHOLD = 20   # Only data with a TBDrms higher than this threshold is used for training
DESCRIPTOR = f"Testing training using new preprocessed dataset - with {NUM_EPOCHS} Epochs"

'''
Loss function options

Options that configure how the loss function is used
'''
PULSE_THRESHOLD = 0.001     # The Pulse is considered to be between the first and last value over the threshold
PENALTY_FACTOR = 500.0      # Values outside the pulse are surpressed, by weighing their error with this factor
WEIGTH_REAL_PART = 5.0      # Weight used for MSE of the real part
WEIGTH_IMAG_PART = 1.0      # Weight used for MSE of the imaginary part
WEIGTH_INTENSITY = 40.0     # Weight used for MSE of the intensity
WEIGTH_PHASE = 5.0          # Weight used for MSE of the phase (only considered, when there is a pulse)
WEIGTH_FROG_ERROR = 0.001   # Weight used for the FROG Error (if it is 0.0, the calculation is skipped)

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
OUTPUT_TIMESTEP = 2*4.89*c.femto                   # Size of timestep between delays [fs]
OUTPUT_START_WAVELENGTH = 489.85*c.nano            # Smallest wavelength in the dataset [nm]
OUTPUT_END_WAVELENGTH = 540.04*c.nano              # Largest wavelength in the dataset [nm]
OUTPUT_START_FREQUENCY = c.c2pi / OUTPUT_START_WAVELENGTH    # convert start wavelength to frequency [Hz]
OUTPUT_END_FREQUENCY = c.c2pi / OUTPUT_END_WAVELENGTH        # convert stop wavelength to frequency [Hz]

'''
Path Options

The paths used are defined here
'''
ModelPath = "./logs/training_002/model.pth"  # path of pretrained model used to initialize weights before training

LogDirectory = "./logs/"
ModelFilename = "model.pth"
LogFilename = "training.log"
LossPlotFilename = "loss.png"
TrainingLossFilename = "training_loss.csv"
ValidationLossFilename = "validation_loss.csv"
LearningRateFilename = "learning_rate.csv"
RandomPredictionFilename = "random_prediction.png"

ModelName = "trained_model_"    # base name of trained models
LogName = "training_"           # base name of logs 
TrainingLossPlotName = "training_loss_" # base name of training loss plots
PredicitonPlotName = "post_training_prediction_"    # base name of prediction plot

# Path = "/mnt/data/desy/frog_simulated/grid_256_v3/" # Path to data used for training 
Path = "/mnt/data/desy/dataset/dataset_01/preproc/simulated/" # Path to data used for training 
TBDrmsFilename = "/mnt/data/desy/dataset/dataset_01/TBD_rms.csv"     # Path to a sorted list of all directories and their corresponfing TBDrms
ShgFilename = "as_gn00.dat"    # Filename of the file containing the SHG-matrix
LabelFilename = "Es.dat"        # Filename of the file containing the label


logger = logging.getLogger(__name__)
'''
getFilepaths()
Search for the highest log subdirectory index and create new directory. Return a list of paths from the input list
Input:
    root_directory      -> directory containing all log subdirectories [string]
    list_files          -> list of strings containing all file names to be saved in the subdirectory [list(string)]
Output:
    list_paths          -> list of strings containing all paths for the files in the subdirectory [list(string)]
'''
def getFilepaths(root_directory, list_files):
    # initialize list of subdirectories
    subdirs = []

    # Loop through all items in the root directory
    for item in os.listdir(root_directory):
        # get the full path of the item
        item_path = os.path.join(root_directory, item)
        # check if the item is a directory that starts with 'training_'
        if os.path.isdir(item_path) and item.startswith("training_"):
            # check if the directory is not empty
            if len(os.listdir(item_path)) != 0:
                # add item to the list of directories
                subdirs.append(item)
            else:
                # do nothing
                pass
        else:
            # do nothing
            pass
    
    # check if subdirs_list is empty
    if not subdirs:
        # No "training_xxx" directory exists, set index to 1
        next_index = 1
    else:
        # sort directories by their index and get the highest one
        max_index = max([
            int(item.split('_')[1]) for item in subdirs     # split the items in subdirs after the '_' to get the index
            ])
        # increment next index
        next_index = max_index + 1

    # create subdirectory name
    next_subdir_name = f"training_{next_index:03d}"     # use leading zeros for consistency
    next_subdir_path = os.path.join(root_directory, next_subdir_name)
    
    # create the next subdirectory
    os.makedirs(next_subdir_path, exist_ok=True)

    list_paths = [
            os.path.join(next_subdir_path, filename) for filename in list_files
            ]

    return list_paths

####################
## get file paths ##
####################
# create a list of filenames
list_filenames = [
        ModelFilename, 
        LogFilename, 
        TrainingLossFilename, 
        ValidationLossFilename,
        LossPlotFilename, 
        RandomPredictionFilename, 
        LearningRateFilename
        ]

# save the filepaths into a list
list_filepaths = getFilepaths(root_directory=LogDirectory, list_files=list_filenames)

# unpack the filepath list
model_filepath,             \
log_filepath,               \
training_loss_filepath,     \
validation_loss_filepath,   \
loss_plot_filepath,         \
random_prediction_filepath, \
learning_rate_filepath,     \
= list_filepaths
