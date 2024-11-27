'''
datasetInformation.py Script
 
Script to get dataset information
'''
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
Define Constants
'''
ROOT_DIRECTORY = "./"
DATA_DIRECTORY = "/mnt/data/desy/frog_simulated/grid_256_v3/"
FILENAME_TBDRMS = "TBDrms_grid_256_v3.csv"

'''
getTBDrmsValues

Calculate TBDrms values for all subdirectories, sort them and write them to a file
'''
def getTBDrmsValues(data_directory, root_directory, output_filename):
    '''
    Inputs:
        data_directory  -> base directorie of the training dataset [string]
        root_directory  -> root directorie of the project [string]
        output_filename -> filename to write the sorted TBDrms values to [string]
    Outputs:
        NULL
    '''
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
    logger.info('Creating sorted CSV-file')
    sorted_data = sorted(data, key=lambda x: x[3], reverse=False)

    sorted_csv_file = os.path.join(root_directory, output_filename)
    with open(sorted_csv_file, 'w', newline='') as sorted_csvfile:
        writer = csv.writer(sorted_csvfile)
        # write header
        writer.writerow(['Directory', 'rmsT', 'rmsW', 'TBDrms'])
        # write data 
        writer.writerows(sorted_data)

'''
processHeader()

Extract the header information from the first two lines of the file.
If the first line has fewer than 5 elements, the header is 2 lines.
Otherwise, it's just the first line.
'''
def processHeader(lines):
    """
    Inputs:
        lines   -> the first two lines of the spectrogram file
    Outputs:
        header  -> the header of the spectrogram file
    """
    header_line_1 = lines[0].split()
    
    if len(header_line_1) < 5:
        header_line_2 = lines[1].split()
        header = header_line_1 + header_line_2
    else:
        header = header_line_1

    if len(header) < 5:
        print("ERROR")
    
    # Convert elements to float for numeric calculations
    header = [float(x) for x in header]
    
    return header

'''
getDatasetInformation()

Description:
    Get the smallest and largest delay and wavelength values from the dataset
'''
def getDatasetInformation(data_directory, matrix_filename=None):
    '''
    Inputs:
        data_directory  -> Directory which contains the data subdirectories [string]
    Outputs:
        min_delay       -> [float] Minimum delay value in dataset
        max_delay       -> [float] Maximum delay value in dataset
        min_wavelength  -> [float] Minumum wavelength value in dataset
        max_wavelength  -> [float] Maximum wavelength value in dataset
    '''

    min_delay = float('inf')
    max_delay = float('-inf')
    min_wavelength = float('inf')
    max_wavelength = float('-inf')

    entries = os.listdir(data_directory)
    
    # check if all entries in the directory are files
    if all(os.path.isfile(os.path.join(data_directory, entry)) for entry in entries):
        for index, file_name in enumerate(entries):
            file_path = os.path.join(data_directory, entry)
            # get the delay, highest and lowest wavelength
            delay, wavelength_lowest, wavelength_highest = getDelayWavelengthFromFile(file_path)
            # Update the minimum timestep
            if delay < max_delay:
                min_delay = delay                    
            # Update the maximum timestep
            if delay > min_delay:
                max_delay = delay
                
            # Update maximum and minimum wavelengths
            if wavelength_highest > max_wavelength:
                max_wavelength = wavelength_highest
            if wavelength_lowest < min_wavelength:
                min_wavelength = wavelength_lowest

    # check if all entries in the directory are subdirectories
    elif all(os.path.isdir(os.path.join(data_directory, entry)) for entry in entries):
        # itterate over all subdirectories
        for subdirectory in enumerate(entries):
            # get the subdirectory path and also the filepath
            subdirectory_path = os.path.join(data_directory, subdirectory)
            file_path = os.path.join(subdirectory_path, matrix_filename)
            # get the delay, highest and lowest wavelength
            delay, wavelength_lowest, wavelength_highest = getDelayWavelengthFromFile(file_path)
            # Update the minimum timestep
            if delay < max_delay:
                min_delay = delay                    
            # Update the maximum timestep
            if delay > min_delay:
                max_delay = delay
                
            # Update maximum and minimum wavelengths
            if wavelength_highest > max_wavelength:
                max_wavelength = wavelength_highest
            if wavelength_lowest < min_wavelength:
                min_wavelength = wavelength_lowest
    else:
        raise ValueError(f"The directory '{data_directory}' contains a mix of files and subdirectories or is empty")
                            
    # Print the results
    logger.info(f"Min Delay: {min_delay}")
    logger.info(f"Max Delay: {max_delay}")
    logger.info(f"Min Wavelength: {min_wavelength}")
    logger.info(f"Max Wavelength: {max_wavelength}")

    return min_delay, max_delay, min_wavelength, max_wavelength

def getDelayWavelengthFromFile(path):

    if not os.path.exists(path):
        raise OSError(f"The file '{path}' does not exist!")

    with open(path, 'r') as file:
        # Read the first two lines
        lines = [file.readline(), file.readline()]
    
        # Extract the header
        header = processHeader(lines)
    
        # Extract the required values
        delay = header[2]  # Third element
        number_wavelength = header[1]  # Second element
        wavelength_step = header[3]  # Fourth element
        center_wavelenght = header[4]  # Fifth element

        # Calculate wavelength related values
        wavelength_range = (number_wavelength // 2) * wavelength_step
        wavelength_highest = center_wavelenght + wavelength_range
        wavelength_lowest = center_wavelenght - wavelength_range

        return delay, wavelength_highest, wavelength_lowest

'''
Calling the functions
'''
getTBDrmsValues(
        data_directory = DATA_DIRECTORY,
        root_directory = ROOT_DIRECTORY,
        output_filename = FILENAME_TBDRMS,
        )
