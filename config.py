import logging
import numpy as np
import os

OUTPUT_SIZE = 128
BATCH_SIZE = 10
NUM_EPOCHS = 10
UNFREEZE_EPOCH = 8
LEARNING_RATE = 0.001
TRAINING_LOG_STEP_SIZE = BATCH_SIZE
# NUMBER_FOLDS = 5
# NUMBER_FOLDS = 2

PENALTY_FACTOR = 1
PENALTY_THRESHOLD = 1
MAX_INTENSITY = 1
MAX_PHASE = 1
# MAX_INTENSITY = 1
# MAX_PHASE = 1

OUTPUT_NUM_DELAYS = 512 
OUTPUT_NUM_WAVELENGTH = 512 
OUTPUT_TIMESTEP = 2.2    # [fs]
OUTPUT_START_WAVELENGTH = 450   # [nm]
OUTPUT_END_WAVELENGTH = 600     # [nm]

ModelDirectory = "./models/"
LogDirectory = "./logs/"
ModelName = "trained_model_"
ModelPath = "./models/trained_model_1.pth"
LogName = "training_"
TrainingLossPlotName = "training_loss_"

# Path = "/mnt/data/desy/frog_simulated/grid_512_v2/"
# Path = "/mnt/data/desy/frog_simulated/grid_256/"
Path = "/mnt/data/desy/frog_simulated/grid_256_v3/"
SpecFilename = "as_gn00.dat"
# SpecFilename = "as.dat"
LabelFilename = "Es.dat"


logger = logging.getLogger(__name__)

'''
getFilepathIndex()
Search the model and log directories for the highest index and determine the next filenames
Input:
    model_directory         -> directory that contains the trained models
    log_directory           -> directory that contains the log files and loss plots
    model_base_filename     -> filename of saved models without index and file ending
    log_base_filename       -> filename of logs without index and file ending
    loss_plot_base_filename -> filename of loss plots without index and file ending
Output:
    model_filepath      -> filepath of the next model file to be saved
    log_filepath        -> filepath of the next log file to be saved
    loss_plot_filepath  -> filepath of the next loss plot file to be saved
'''
def getFilepathIndex(model_directory, log_directory, model_base_filename, log_base_filename, loss_plot_base_filename):

    # Check if log and model directory exists, if not create them
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    # Get a list of all files in the directory
    model_files = os.listdir(model_directory)
    log_files = os.listdir(log_directory)

    # Filter out matching file names and write them in a list
    matching_model_files = [f for f in model_files if f.startswith(model_base_filename) and f.endswith('.pth')]
    matching_loss_plot_files = [f for f in log_files if f.startswith(loss_plot_base_filename) and f.endswith('.png')]
    matching_log_files= [f for f in log_files if f.startswith(log_base_filename) and f.endswith('.png')]

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

    # get the largest index
    next_index = max(next_model_index, next_loss_plot_index, next_log_index)

    # Get the new filename
    new_model_filename = f"{model_base_filename}{next_index}.pth"
    new_loss_plot_filename = f"{loss_plot_base_filename}{next_index}.png"
    new_log_filename = f"{log_base_filename}{next_index}.log"

    # Join path and filenames together
    model_filepath = os.path.join(model_directory, new_model_filename)
    loss_plot_filepath = os.path.join(log_directory, new_loss_plot_filename)
    log_filepath = os.path.join(log_directory, new_log_filename)

    return model_filepath, log_filepath, loss_plot_filepath

####################
## get file paths ##
####################
model_filepath, log_filepath, loss_plot_filepath = getFilepathIndex(
        model_directory=ModelDirectory,
        log_directory=LogDirectory,
        model_base_filename=ModelName,
        log_base_filename=LogName,
        loss_plot_base_filename=TrainingLossPlotName
)
