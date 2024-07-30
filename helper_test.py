'''
HELPER FUNCTIONS
'''
#############
## Imports ##
#############
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class TimeDomain:
    def __init__(self, time_axis, intensity, phase, real, imag):
        self.time_axis = time_axis
        self.intensity = intensity
        self.phase = phase
        self.real = real
        self.imag = imag

'''
RESAMPLE SPECTROGRAMS
'''
class ResampleSpectrogram(object):

    def __init__(self, num_delays_out, timestep_out, num_wavelength_out, start_wavelength_out, end_wavelength_out):
        output_number_rows = num_delays_out
        output_time_step = timestep_out
        output_number_cols = num_wavelength_out
        output_start_wavelength = start_wavelength_out   # the first element of the wavelength output axis
        output_stop_wavelength = end_wavelength_out    # the last element of the wavelength output axis 
        output_start_time = -int(output_number_rows/2) * output_time_step     # calculate time at which the output time axis starts
        output_end_time = output_start_time + (output_time_step * output_number_rows) - output_time_step    # calculate the last element of the output time axis 

        ######################## Used here to save time later
        ## Define output axis ##
        ########################
        self.output_time = np.linspace(output_start_time, output_end_time, output_number_rows )  # create array that corresponds to the output time axis
        self.output_wavelength = np.linspace(output_start_wavelength, output_stop_wavelength, output_number_cols)    # create array that corresponds tot the output wavelength axis
 

    def __call__(self, path):
        # Constants 
        NUM_HEADER_ELEMENTS = 5

        #########################
        ## Read Header of file ##
        #########################
        with open(path, mode='r') as f:     # open the file in read mode
            first_line = f.readline().strip('\n')   # read the first line
            second_line = f.readline().strip('\n')  # read the second line
        
        first_line_len = len(first_line.split() ) # number of elements in the first line
        second_line_len = len(second_line.split() ) # number of elements in the second line
        
        # check if 1st and 2nd line have 5 Elements in total
        if(first_line_len + second_line_len != NUM_HEADER_ELEMENTS):
            # check if 1st line has 5 Elements in total
            if(first_line_len != NUM_HEADER_ELEMENTS):
                # the file has the wrong format
                print(f'Error: Number of Header Elements != {NUM_HEADER_ELEMENTS}')
                return
            else:
                # fist line has 5 Elements -> write into header
                header = first_line.split()
                num_rows_skipped = 1 # header is in first row
                # print("Header is in 1 rows")    # for debugging
        else:
            # first 2 lines have 5 Elements in total -> write into header
            header = first_line.split() + second_line.split()
            num_rows_skipped = 2 # header is in first 2 rows
            # print("Header is in 2 rows")    # for debugging
        
        input_number_rows = int(header[0]) # delay points
        input_number_cols = int(header[1]) # wavelength points
        input_time_step = int(header[2]) # [fs]
        input_wavelength_step = float(header[3]) # [nm]
        input_center_wavelength = float(header[4]) # [nm]
        
        ######################
        ## Read Spectrogram ##
        ######################
        spectrogram_df = pd.read_csv(path, sep='\\s+', skiprows=num_rows_skipped, header=None)
        spectrogram = spectrogram_df.to_numpy()     # convert to numpy array 
        
        #######################
        ## Define input axis ##
        #######################
        # calculate input time axis
        input_start_time = -int(input_number_rows / 2) * input_time_step    # calculate time at which the input time axis starts
        input_end_time = input_start_time + (input_time_step * input_number_rows) - input_time_step    # calculate the last element of the input time axis
        input_time = np.linspace(input_start_time, input_end_time, input_number_rows)   # create array that corresponds to the input time axis
        
        # calculate input wavelength axis
        input_start_wavelength = input_center_wavelength - (input_number_cols/2)*input_wavelength_step      # calculate the first element of the wavelength input axis
        input_stop_wavelength = input_center_wavelength + (input_number_cols/2)*input_wavelength_step       # calculate the last element of the wavelength input axis
        input_wavelength = np.linspace(input_start_wavelength, input_stop_wavelength, input_number_cols)    # create array that corresponds tot the input wavelength axis
        
        ##############################
        ## Resample wavelength axis ##
        ##############################
        interpolate_wavelength = interp1d(input_wavelength, spectrogram, axis=1, kind='linear', bounds_error=False, fill_value=0) 
        output_spectrogram = interpolate_wavelength(self.output_wavelength)
        
        ########################
        ## Resample time axis ##
        ######################## 
        interpolate_time = interp1d(input_time, output_spectrogram, axis=0, kind='linear', bounds_error=False, fill_value=0)
        output_spectrogram = interpolate_time(self.output_time)
        
        return spectrogram, input_time, input_wavelength, output_spectrogram, self.output_time, self.output_wavelength
 
'''
READ Labels (Real and Imaginary Part) from Es.dat
'''
class ReadLabelFromEs(object):
    def __init__(self):
        pass

    def __call__(self, path):    
        # Read Ek.dat file and place columns in arrays
        # Inputs:
        # Path = Path to Ek.dat
        # Outputs:
        # time_axis = Array containing time axis of time signal
        # intensity = Array containing intensity of time signal (squared amplitude)
        # phase = Array containing phase of time signal
        # real_part = Array containing real part of time signal
        # imag_part = Array containing imaginary part of time signal

        # read the dataframe
        dataframe = pd.read_csv(path,sep='  ', header=None)     # sep needs to be 2 spaces
        
        TimeDomainSignal = TimeDomain(time_axis = dataframe[0].to_numpy(),
                                      intensity = dataframe[1].to_numpy(),
                                      phase = dataframe[2].to_numpy(), 
                                      real = dataframe[3].to_numpy(),
                                      imag = dataframe[4].to_numpy())
        label = np.concatenate( (TimeDomainSignal.real, TimeDomainSignal.imag), axis=0)
        return label

'''
Compare Spectrograms
'''
def compareSpectrograms(original_spectrogram, original_time_axis, original_wavelength_axis,
                        resampled_spectrogram, resampled_time_axis, resampled_wavelength_axis):
    # Plots original and resampled spectrograms
    # Inputs:
    # original_spectrogram = non-resampled spectrogram
    # original_time_axis = array containing time axis of non-resampled spectrogram
    # original_wavelength_axis = array containing wavelength axis of non-resampled spectrogram
    # resampled_spectrogram = resampled spectrogram
    # resamplel_time_axis = array containing time axis of resampled spectrogram
    # resamplel_wavelength_axis = array containing wavelength axis of resampled spectrogram
    # Outputs:
    # NULL

    plt.figure()    # create figure
    plt.subplot(1,2,1)  # create first subplot
    # create plot of original spectrogram
    plt.imshow(original_spectrogram,
               aspect='auto',
               # aspect='equal', 
               extent=[original_wavelength_axis[0],
                       original_wavelength_axis[-1],
                       original_time_axis[0],
                       original_time_axis[-1]],
               origin='lower',
               cmap='viridis' )
    plt.title('Original Spectrogram')
    plt.ylabel('Time [fs]')
    plt.xlabel('Wavelength [nm]')

    plt.subplot(1, 2, 2)    # create second subplot
    # create plot of resampled spectrogram
    plt.imshow(resampled_spectrogram, 
               aspect='auto',
               # aspect='equal', 
               extent=[resampled_wavelength_axis[0],
                       resampled_wavelength_axis[-1],
                       resampled_time_axis[0],
                       resampled_time_axis[-1]],
               origin='lower',
               cmap='viridis' ) 
    plt.title('Resampled Spectrogram')
    plt.ylabel('Time [fs]')
    plt.xlabel('Wavelength [nm]')

    plt.tight_layout() # Place spectrograms close together
    plt.show()  # show figure
'''
Plot Time Signals
'''
def plotTimeDomain(TimeDomain, TimeDomainLabel):
    print(f'Time {TimeDomain.time_axis[257]}')
    print(f'Intensity {TimeDomain.intensity[257]}')
    print(f'Phase {TimeDomain.phase[257]}')
    print(f'Real {TimeDomain.real[257]}')
    print(f'Imag {TimeDomain.imag[257]}')
    fig, ax = plt.subplots()    # create figure
    plt.subplot(1,2,1)  # create first subplot

    # create plot of original spectrogram
    ax.plot(TimeDomain.time_axis, np.sqrt(TimeDomain.intensity), 'c')
    ax.plot(TimeDomain.time_axis, TimeDomain.real, 'r', ls=':')
    ax.plot(TimeDomain.time_axis, TimeDomain.imag, 'b-', ls=':')
    ax.plot(TimeDomain.time_axis, TimeDomain.phase, 'g')
    ax.title('Time Domain Signal')
    ax.xlabel('Time [fs]')
    ax.ylabel('Intensity')

    plt.tight_layout() # Place plots close together
    plt.show()  # show figure
    
'''
Calculate Time Domain Values
'''
def calcTimeDomain(Real, Imag):
    # Calculates Intensity and Phase from real and imaginary part of time signal
    # Inputs: 
    # Real = Real part of time signal
    # Imag = Imaginary part of time signal
    # Outputs: 
    # Intensity = Intensity of time signal (squared amplitude) 
    # Phase = Phase of time signal

    #########################
    ## Calculate Intensity ##
    #########################
    # Intensity = Real^2 + Imag^2
    Intensity = Real**2+ Imag**2

    #####################
    ## Calculate Phase ##
    #####################
    # Phase = arctan(Imag/Real)
    Phase = np.arctan(Imag/Real)

    return Intensity, Phase

'''
Read Ek.dat
'''
# TODO test
def readEkDat(Path):
    # Read Ek.dat file and place columns in arrays
    # Inputs:
    # Path = Path to Ek.dat
    # Outputs:
    # time_axis = Array containing time axis of time signal
    # intensity = Array containing intensity of time signal (squared amplitude)
    # phase = Array containing phase of time signal
    # real_part = Array containing real part of time signal
    # imag_part = Array containing imaginary part of time signal

    # read the dataframe
    dataframe = pd.read_csv('Ek.dat',sep='  ', header=None)     # sep needs to be 2 spaces
    
    TimeDomainSignal = TimeDomain(time_axis = dataframe[0].to_numpy(),
                                  intensity = dataframe[1].to_numpy(),
                                  phase = dataframe[2].to_numpy(), 
                                  real = dataframe[3].to_numpy(),
                                  imag = dataframe[4].to_numpy())
    # place each column into its own numpy array
    # time_axis   = dataframe[0].to_numpy()
    # intensity   = dataframe[1].to_numpy()
    # phase       = dataframe[2].to_numpy()
    # real_part   = dataframe[3].to_numpy()
    # imag_part   = dataframe[4].to_numpy()

    # return the arrays
    # return time_axis, intensity, phase, real_part, imag_part
    return TimeDomainSignal

'''
Create Ek.dat
'''
# TODO test
# def createEkDat(Path, time_axis, intensity, phase, real_part, imag_part):
def createEkDat(Path, TimeDomainSignal):
    # Place time signal into Ek.dat file
    # Inputs:
    # Path = Path to Ek.dat
    # time_axis = Array containing time axis of time signal
    # intensity = Array containing intensity of time signal (squared amplitude)
    # phase = Array containing phase of time signal
    # real_part = Array containing real part of time signal
    # imag_part = Array containing imaginary part of time signal
    # Outputs:
    # NULL
    ts = TimeDomainSignal
    # Create a dataframe and placing each input array in one column
    # data = np.column_stack((time_axis, intensity, phase, real_part, imag_part))
    data = np.column_stack((ts.time_axis, ts.intensity, ts.phase, ts.real_part, ts.imag_part))
    # Save the dataframe into a file at the specified path
    np.savetxt(Path, data, fmt='%f', delimiter=' ', comments='')
