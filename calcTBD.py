import os
import csv
import logging

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
GetTBDrmsValues
'''
def GetTBDrmsValues(data_directory, root_directory, output_filename, sorted_output_filename):
    data = []
    
    logger.info('Stepping through all subdirectories')
    # step through all directories
    for subdirectory in os.listdir(data_directory):
        # get the subdirectory path 
        subdirectory_path = os.path.join(data_directory, subdirectory)

        # Ensure it's a subdirectory and starts with 's'
        if os.path.isdir(subdirectory_path) and subdirectory.startswith('s'):
            # get the path of SimulatedPulseData.txt
            file_path = os.path.join(subdirectory_path, 'SimulatedPulseData.txt')

            # Check if 'SimulatedPulseData.txt' exists
            if os.path.exists(file_path):
                # initialize rmsT and rmsW 
                rmsT, rmsW = None, None

                # open and read 'SimulatedPulseData.txt'
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        # Extract rmsT value
                        if line.startswith('rmsT;'):
                            rmsT = float(line.split(';')[1].strip())
                        # Extract rmsW value
                        if line.startswith('rmsW;'):
                            rmsW = float(line.split(';')[1].strip())

                # Check if rmsT and rmsW have values
                if (rmsT is not None) and (rmsW is not None):
                    TBDrms = rmsT*rmsW
                    
                    # Store values as a tuple (subdirectory, rmsT, rmsW, TBDrms)
                    data.append([subdirectory, rmsT, rmsW, TBDrms])
    logger.info('Got all data from subdirectories')
    #####################
    ## WRITE CSV FILES ##
    #####################
    logger.info('Creating first CSV-file')
    csv_file = os.path.join(root_directory, output_filename)
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # write header
        writer.writerow(['Directory', 'rmsT', 'rmsW', 'TBDrms'])
        # write data 
        writer.writerows(data)

    logger.info('Creating second CSV-file')
    sorted_data = sorted(data, key=lambda x: x[3], reverse=False)

    sorted_csv_file = os.path.join(root_directory, sorted_output_filename)
    with open(sorted_csv_file, 'w', newline='') as sorted_csvfile:
        writer = csv.writer(sorted_csvfile)
        # write header
        writer.writerow(['Directory', 'rmsT', 'rmsW', 'TBDrms'])
        # write data 
        writer.writerows(sorted_data)

'''
Calling the function
'''
ROOT_DIRECTORY = "./"
DATA_DIRECTORY = "/mnt/data/desy/frog_simulated/grid_256_v3/"
FILENAME = "TBDrms_list_grid_256_v3.csv"
FILENAME_SORTED = "TBDrms_list_grid_256_v3_sorted.csv"

GetTBDrmsValues(
        data_directory = DATA_DIRECTORY,
        root_directory = ROOT_DIRECTORY,
        output_filename = FILENAME,
        sorted_output_filename = FILENAME_SORTED
        )