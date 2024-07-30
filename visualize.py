import matplotlib.pyplot as plt
import numpy as np

def visualize(spectrogram, label, prediction):

    real = label[:,1]
    imag = label[:,2]
    abs = np.abs(real + 1j* imag) 
   
    fig = plt.figure() 
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)  # colspan=3 means the plot spans 3 columns
    ax1.plot(label[:,0], abs)
    ax1.set_title('Zeitsignal')
    if((prediction) == len(spectrogram)):
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
