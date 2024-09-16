import matplotlib.pyplot as plt
import numpy as np

'''
ScaleLabel()
Scales Labels to range [-1,1]
'''
class ScaleLabel(object):
    def __init__(self, max_intensity, max_phase):
        '''
        Inputs:
            max_intensity   -> >= maximum intesity in dataset [float]
            max_phase       -> >= maximum phase in dataset [float]
        '''
        self.max_intensity = max_intensity
        self.max_phase = max_phase

    def __call__(self, label):    
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase [tensor]
        Outputs:
            scaled_label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
                scaled to [-1,1] [tensor]
        '''
        length_label = len(label)
        half_size = int(length_label //2)
        intensity = label[:half_size]  # First half -> intensity
        phase = label[half_size:]      # Second half -> phase
        intensity_scaled = intensity / self.max_intensity
        phase_scaled = phase / self.max_phase

        # Concatenate the two halves back together
        scaled_label = torch.cat((intensity_scaled, phase_scaled), dim=0)

        return scaled_label

'''
UnscaleLabel()
Restores original scale of labels
'''
class UnscaleLabel(object):
    def __init__(self, max_intensity, max_phase):
        '''
        Inputs:
            max_intensity   -> >= maximum intesity in dataset (needs to be the same as in ScaleLabel) [float]
            max_phase       -> >= maximum phase in dataset (needst to be the same as in ScaleLabel) [float]
        '''
        self.max_intensity = max_intensity
        self.max_phase = max_phase

    def __call__(self, scaled_label):    
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            scaled_label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
                scaled to [-1,1] [tensor]
        Outputs:
            label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase [tensor]
        '''
        length_label = len(scaled_label)
        half_size = int(length_label //2)
        intensity_scaled = scaled_label[:half_size]  # First half -> intensity
        phase_scaled = scaled_label[half_size:]      # Second half -> phase
        intensity = intensity_scaled * self.max_intensity
        phase = phase_scaled * self.max_phase

        # Concatenate the two halves back together
        label = torch.cat((intensity, phase), dim=0)

        return label

'''
UnscaleLabel()
Restores original scale of labels
'''
class UnscaleLabel(object):
    def __init__(self, max_intensity, max_phase):
        '''
        Inputs:
            max_intensity   -> >= maximum intesity in dataset (needs to be the same as in ScaleLabel) [float]
            max_phase       -> >= maximum phase in dataset (needst to be the same as in ScaleLabel) [float]
        '''
        self.max_intensity = max_intensity
        self.max_phase = max_phase

    def __call__(self, scaled_label):    
        '''
        Scale the values of intensity and phase to [-1,1]
        Inputs:
            scaled_label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase
                scaled to [-1,1] [tensor]
        Outputs:
            label -> List of arrays containing intesity of time signal (squared amplitute) and it's phase [tensor]
        '''
        length_label = len(scaled_label)
        half_size = int(length_label //2)
        intensity_scaled = scaled_label[:half_size]  # First half -> intensity
        phase_scaled = scaled_label[half_size:]      # Second half -> phase
        intensity = intensity_scaled * self.max_intensity
        phase = phase_scaled * self.max_phase

        # Concatenate the two halves back together
        label = torch.cat((intensity, phase), dim=0)

        return label

'''
ResampleSpectrogram()
Transform Class that resamples spectrograms to use the same axis and size
'''
class ResampleSpectrogram(object):

    def __init__(self, num_delays_out, timestep_out, num_wavelength_out, start_wavelength_out, end_wavelength_out):
        '''
        Inputs:
            num_delays_out          -> Number of delay values the resampled spectrogram will have [int]
            timestep_out            -> Length of time between delays [fs] [int]
            num_wavelength_out      -> Number of wavelength values the resampled spectrogram will have [int]
            start_wavelength_out    -> Lowest wavelength value of the resampled spectrogram [nm] [int]
            end_wavelength_out      -> Highest wavelength value of the resampled spectrogram [nm] [int]
        '''
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
        '''
        Takes path of spectrogram and resamples it to the configured size and range of time/wavelength 
        Inputs:
            path    -> Path to spectrogram [string]
        Outputs:
            spectrogram             -> Original (not resampled) spectrogram [tensor]
            input_time              -> Original (not resampled) time axis [numpy array]
            input_wavelength        -> Original (not resampled) wavelength axis [numpy array]
            output_spectrogram      -> Resampled spectrogram [tensor]
            self.output_time        -> Resampled time axis [numpy array]
            self.output_wavelength  -> Resampled wavelenght axis [numpy array]
        '''
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
                logger.error(f"Number of Header Elements != {NUM_HEADER_ELEMENTS}")
                # print(f'Error: Number of Header Elements != {NUM_HEADER_ELEMENTS}')
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
        input_time_step = float(header[2]) # [fs]
        input_wavelength_step = float(header[3]) # [nm]
        input_center_wavelength = float(header[4]) # [nm]
        
        ######################
        ## Read Spectrogram ##
        ######################
        spectrogram_df = pd.read_csv(path, sep='\\s+', skiprows=num_rows_skipped, header=None, engine='python')
        spectrogram = spectrogram_df.to_numpy()     # convert to numpy array 
        spectrogram = torch.from_numpy(spectrogram)     # convert to tensor

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
        output_spectrogram = torch.from_numpy(output_spectrogram)  # convert to tensor
        spectrogram = spectrogram

        return spectrogram, input_time, input_wavelength, output_spectrogram, self.output_time, self.output_wavelength


def visualize(spectrogram, label, prediction):

    real = label[:256]
    imag = label[256:]
    abs = np.abs(real + 1j* imag) 
   
    fig = plt.figure() 
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)  # colspan=3 means the plot spans 3 columns
    ax1.plot(label[:,0], abs)
    ax1.set_title('Zeitsignal')
    if((prediction) == len(label)):
        ax1.plot(label[:,0], prediction)
        ax1.set_title('Vorhergesagtes Zeitsignal')
    else:
        print('Length of Prediction and Label not the same')
    
    
    # Smaller plots in the second row
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax2.imshow(spectrogram[0])
    ax2.set_title('Spectrogram')
    ax2.axis('off')
    
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    ax3.imshow(spectrogram[1])
    ax3.set_title('Time')
    ax3.axis('off')
    
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    ax4.imshow(spectrogram[2])
    ax4.set_title('Frequency')
    ax4.axis('off')
    
    fig.suptitle("Spectrogram, Time and Frequency")

    plt.show()

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


