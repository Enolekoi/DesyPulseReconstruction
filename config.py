import sys
import logging

import helper

LogDirectory = "./logs/"
LogName = "training_"
TrainingLossPlotName = "training_loss_"

log_filepath, loss_plot_filepath = helper.getLogFilepath(
        directory=LogDirectory,
        log_base_filename=LogName,
        loss_plot_base_filename=TrainingLossPlotName
)

'''
Configure Logging
'''
def configure_logging(log_filepath):
    logger = logging.getLogger(__name__)    # create logger with the name of the current module

    logging_console_handler = logging.StreamHandler(sys.stdout)   # create a handler for the console log
    logging_file_handler = logging.FileHandler(         # create a handler for the file log
            log_filepath,
            encoding="utf-8",
            mode="a"
    )
    logging_formatter = logging.Formatter(  # create a formatter
            "{asctime} - {name} - {funcName} - {levelname}: {message}",
            datefmt="%d-%m-%Y %H:%M:%S",
            style="{"
    )
    logging_console_handler.setFormatter(logging_formatter)     # add the formatter to the console log handler
    logging_file_handler.setFormatter(logging_formatter)        # add the formatter to the console file handler
    logging_console_handler.setLevel(logging.DEBUG)     # Set the logger level to debug
    logging_file_handler.setLevel(logging.DEBUG)        # Set the logger level to debug
    logger.addHandler(logging_console_handler)  # add the console log handler to the logger
    logger.addHandler(logging_file_handler)     # add the file log handler to the logger
    
    logger.setLevel(logging.DEBUG)  # Set the logger level to debug
    # logger.setLevel(logging.INFO)   # Set the logger level to info
    return logger

