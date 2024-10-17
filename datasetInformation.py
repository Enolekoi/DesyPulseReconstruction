'''
datasetInformation.py Script
 
Script to get dataset information
'''
import sys
sys.path.append('./modules/')
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
getTimeWavelengthValues()
Process 
'''
def getTimeWavelengthValues(data_directory):
    '''
    Inputs:
        data_directory  -> Directory which contains the data subdirectories [string]
    Outputs:
        NULL
    '''
    max_timestep = float('-inf')
    max_wavelength = float('-inf')
    min_wavelength = float('inf')
    
    for subdirectory in os.listdir(data_directory):
        subdirectory_path = os.path.join(data_directory, subdirectory)
        if os.path.isdir(subdirectory_path):
            file_path = os.path.join(subdirectory_path, 'as_gn00.dat')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    # Read the first two lines
                    lines = [f.readline(), f.readline()]
                    
                    # Extract the header
                    header = processHeader(lines)
                    
                    # Extract the required values
                    timestep = header[2]  # Third element
                    number_wavelength = header[1]  # Second element
                    wavelength_step = header[3]  # Fourth element
                    center_wavelenght = header[4]  # Fifth element

                    # Update the maximum timestep
                    if timestep > max_timestep:
                        max_timestep = timestep
                    
                    # Calculate wavelength related values
                    wavelength_range = (number_wavelength // 2) * wavelength_step
                    wavelength_plus = center_wavelenght + wavelength_range
                    wavelength_minus = center_wavelenght - wavelength_range
                    
                    # Update maximum and minimum wavelengths
                    if wavelength_plus > max_wavelength:
                        max_wavelength = wavelength_plus
                    if wavelength_minus < min_wavelength:
                        min_wavelength = wavelength_minus
                            
    # Print the results
    logger.info(f"Highest Time Step: {max_timestep}")
    logger.info(f"Max Wavelength: {max_wavelength}")
    logger.info(f"Min Wavelength: {min_wavelength}")

'''
Calling the functions
'''
getTBDrmsValues(
        data_directory = DATA_DIRECTORY,
        root_directory = ROOT_DIRECTORY,
        output_filename = FILENAME_TBDRMS,
        )

getTimeWavelengthValues(
        data_directory = DATA_DIRECTORY
        )
