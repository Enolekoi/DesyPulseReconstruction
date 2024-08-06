import matplotlib.pyplot as plt
import numpy as np

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


