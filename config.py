import logging
import os

OUTPUT_SIZE = 256
BATCH_SIZE = 10
NUM_EPOCHS = 2
LEARNING_RATE = 0.0001
NUMBER_FOLDS = 5

OUTPUT_NUM_DELAYS = 512 
OUTPUT_NUM_WAVELENGTH = 512 
OUTPUT_TIMESTEP = 5    # [fs]
OUTPUT_START_WAVELENGTH = 350   # [nm]
OUTPUT_END_WAVELENGTH = 550     # [nm]


LogDirectory = "./logs/"
LogName = "training_"
TrainingLossPlotName = "training_loss_"

Path = "/mnt/data/desy/frog_simulated/grid_256/"
SpecFilename = "as.dat"
LabelFilename = "Es.dat"

logger = logging.getLogger(__name__)
'''
get Log Filepaths
'''
def getLogFilepath(directory, log_base_filename, loss_plot_base_filename):

    # Check if directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Only filter out matching file names
    matching_loss_plot_files = [f for f in files if f.startswith(loss_plot_base_filename) and f.endswith('.png')]
    matching_log_files= [f for f in files if f.startswith(log_base_filename) and f.endswith('.png')]

    # Determin what the next loss plot index is
    if matching_loss_plot_files:
        number = numbers = [int(f[len(loss_plot_base_filename):-4]) for f in matching_loss_plot_files if f[len(loss_plot_base_filename):-4].isdigit()]
        next_loss_plot_index = max(numbers) + 1 if numbers else 1
    else:
        next_loss_plot_index = 1

    # Determin what the next loss plot index is
    if matching_log_files:
        number = numbers = [int(f[len(log_base_filename):-4]) for f in matching_log_files if f[len(log_base_filename):-4].isdigit()]
        next_log_index = max(numbers) + 1 if numbers else 1
    else:
        next_log_index = 1

    # get the largest index
    next_index = max(next_loss_plot_index, next_log_index)

    # Get the new filename
    new_loss_plot_filename = f"{loss_plot_base_filename}{next_index}.png"
    new_log_filename = f"{log_base_filename}{next_index}.log"
    loss_plot_filepath = os.path.join(directory, new_loss_plot_filename)
    log_filepath = os.path.join(directory, new_log_filename)

    return log_filepath, loss_plot_filepath

####################
## get file paths ##
####################
log_filepath, loss_plot_filepath = getLogFilepath(
        directory=LogDirectory,
        log_base_filename=LogName,
        loss_plot_base_filename=TrainingLossPlotName
)
