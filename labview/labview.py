'''
labview.py Module 

Module containing functions to be integrated into the labview programm
'''
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import os
# add path of parent directory, so that modules from the modules package can be used from labview
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import constants as c
from modules import preprocessing

'''
plotSpectrogram

Plot a spectrogram using matplotlib
'''
def predictTimeDomain(header_string, shg_matrix):
    '''
    Inputs:
        header_string   -> header of the spectrogram as a string
        shg_matrix      -> Matrix containing spectrogram to be plotted 
    Outputs:
        finished        -> Variable containing information if function is finished [boolean]
    '''
    # get the header and create axes
    header = getHeaderFromString(header_string)
    delay_axis, wavelength_axis = preprocessing.generateAxes(header)
    
    # convert shg_matrix to tensor
    shg_matrix = torch.tensor(shg_matrix)
    # convert shg_matrix to numpy array for plotting
    shg_matrix = shg_matrix

    plt.ioff()    
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    # plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()], cmap='viridis')
    plt.pcolormesh(
            delay_axis.numpy(),
            wavelength_axis.numpy(),
            shg_matrix.numpy().T,
            shading='auto'
            )

    # Add labels and title
    plt.colorbar(label='Intensity')  # or the appropriate unit for your data
    plt.ylabel("Wavelength [nm]")
    plt.xlabel("Time [fs]")
    plt.title(f"SHG Matrix; Header: {header}")

    # Show the plot
    plt.show()
    
    finished = True

    return finished

def getHeaderFromString(header_string):
    # split the string into a list
    header_list = header_string.split()
    
    # make the first two elements of the header_list integers and the last three floats
    # delta_tau to fs
    # delta_lambda, lambdaCenter to nm
    header_list = [
            int(header_list[0]),
            int(header_list[1]),
            float(header_list[2]) * c.femto, 
            float(header_list[3]) * c.nano,
            float(header_list[4]) * c.nano
            ]
    # convert the header_list to a tensor
    header = torch.tensor(header_list)

    return header
