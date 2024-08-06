import sys
import logging
import os

LogDirectory = "./logs/"
LogName = "training_"
TrainingLossPlotName = "training_loss_"

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

# '''
# Configure Logging
# '''
# def configure_logging(log_filepath):
#     logger = logging.getLogger(__name__)    # create logger with the name of the current module

#     logging_console_handler = logging.StreamHandler(sys.stdout)   # create a handler for the console log
#     logging_file_handler = logging.FileHandler(         # create a handler for the file log
#             log_filepath,
#             encoding="utf-8",
#             mode="a"
#     )
#     logging_formatter = logging.Formatter(  # create a formatter
#             "{asctime} - {name} - {funcName} - {levelname}: {message}",
#             datefmt="%d-%m-%Y %H:%M:%S",
#             style="{"
#     )
#     logging_console_handler.setFormatter(logging_formatter)     # add the formatter to the console log handler
#     logging_file_handler.setFormatter(logging_formatter)        # add the formatter to the console file handler
#     logging_console_handler.setLevel(logging.DEBUG)     # Set the logger level to debug
#     logging_file_handler.setLevel(logging.DEBUG)        # Set the logger level to debug
#     logger.addHandler(logging_console_handler)  # add the console log handler to the logger
#     logger.addHandler(logging_file_handler)     # add the file log handler to the logger
#     
#     logger.setLevel(logging.DEBUG)  # Set the logger level to debug
#     # logger.setLevel(logging.INFO)   # Set the logger level to info
#     return logger

####################
## get file paths ##
####################
log_filepath, loss_plot_filepath = getLogFilepath(
        directory=LogDirectory,
        log_base_filename=LogName,
        loss_plot_base_filename=TrainingLossPlotName
)
