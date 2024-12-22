'''
datasetInformation.py Script
 
Script to get dataset information
'''
import os
import csv
import logging

from modules import preprocessing as pre
from modules import helper

'''
Initialize logger
'''
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

'''
Define Constants
'''
GRID_SIZE = 256
ROOT_DIRECTORY = "./"
DATA_DIRECTORY = "/mnt/data/desy/frog_simulated/grid_256_v3/"
MATRIX_FILENAME= "as_gn00.dat"
FILENAME_TBDRMS = "grid_256_v4_TBD.csv"
FILENAME_DATASET_INFO = os.path.join(ROOT_DIRECTORY, "grid_256_v4_info.txt")

# get number of experimental datapoints
num_directories, num_files = helper.countFilesAndDirectories(directory_path=DATA_DIRECTORY)
num_elements = num_directories + num_files
logger.info(f"Total amount of directories in '{DATA_DIRECTORY}': {num_directories}")
logger.info(f"Total amount of files in       '{DATA_DIRECTORY}': {num_files}")
logger.info(f"Total amount of elements in    '{DATA_DIRECTORY}': {num_elements}\n")

logger.info("Getting Dataset Information")
min_delay, max_delay, min_wavelength, max_wavelength = pre.getDatasetInformation(
    data_directory=DATA_DIRECTORY,
    matrix_filename=MATRIX_FILENAME
    )

logger.info("DELAYS")
logger.info(f"Min. Delay        : {min_delay} fs")
logger.info(f"Max. Delay        : {max_delay} fs")
logger.info(f"Min. delta tau    : {min_delay/(GRID_SIZE//2)} fs ({GRID_SIZE} grid)")
logger.info(f"Max. delta tau    : {max_delay/(GRID_SIZE//2)} fs ({GRID_SIZE} grid)\n")
logger.info("WAVELENGTH")
logger.info(f"Min. Wavelength   : {min_wavelength} nm")
logger.info(f"Max. Wavelength   : {max_wavelength} nm")
logger.info(f"Min. delta lambda : {min_delay/(GRID_SIZE)} nm ({GRID_SIZE} grid)")
logger.info(f"Max. delta lambda : {min_delay/(GRID_SIZE)} nm ({GRID_SIZE} grid)\n")

logger.info(f"Writing information into '{FILENAME_DATASET_INFO}'!")
# write to info file for raw simulated data
pre.writeDatasetInformationToFile(        
        file_path = FILENAME_DATASET_INFO,
        dataset_path = DATA_DIRECTORY,
        num_datapoints = num_directories,
        min_delay = min_delay,  
        max_delay = max_delay, 
        min_wavelength = min_wavelength, 
        max_wavelength = max_wavelength, 
        grid_size = GRID_SIZE)

logger.info(f"Writing TBD_rms values for each Datapoint into'{FILENAME_TBDRMS}'!")
pre.getTBDrmsValues(
    data_directory = DATA_DIRECTORY,
    root_directory = ROOT_DIRECTORY, 
    output_filename = FILENAME_TBDRMS
    )
