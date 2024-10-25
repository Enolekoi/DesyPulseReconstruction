'''
labview.py Module 

Module containing functions to be integrated into the labview programm
'''
import matplotlib.pyplot as plt

'''
plotSpectrogram

Plot a spectrogram using matplotlib
'''
def predict_time_domain(header, shg_matrix):
    '''
    Inputs:
        header          -> header of the spectrogram
        shg_matrix      -> Matrix containing spectrogram to be plotted 
    Outputs:
        finished        -> Variable containing information if function is finished [boolean]
    '''
    finished = False
    plt.ioff()    
    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    # plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[t.min(), t.max(), f.min(), f.max()], cmap='viridis')
    plt.imshow(shg_matrix, aspect='auto', origin='lower', cmap='viridis')

    # Add labels and title
    plt.colorbar(label='Intensity')  # or the appropriate unit for your data
    plt.ylabel("Wavelength [nm]")
    plt.xlabel("Time [fs]")
    plt.title(f"SHG Matrix; Header: {header}")

    # Show the plot
    plt.show()
    
    finished = True

    return finished
