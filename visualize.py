import matplotlib.pyplot as plt
import numpy as np

def compareTimeDomain(filepath, label, prediction):
    orig_real = label[:256]
    orig_imag = label[256:]
    pred_real = prediction[:256]
    pred_imag = prediction[256:]
    
    orig_abs = np.abs(orig_real + 1j* orig_imag) 
    pred_abs = np.abs(pred_real + 1j* pred_imag)

    fig, axs = plt.subplots(4,1)
    # Plotting the Intensity
    axs[0].plot(orig_abs, label="original pulse", color="green")
    axs[0].plot(pred_abs, label="predicted pulse", color="red")
    axs[0].set_ylabel("Intensity of E-Field")
    axs[0].set_xlabel("Time in fs")
    axs[0].grid(True)
    axs[0].legend()

    # Plotting the Real Part
    axs[1].plot(orig_real, label="original pulse", color="green")
    axs[1].plot(pred_real, label="predicted pulse", color="red")
    axs[1].set_ylabel("Real Part of E-Field")
    axs[1].set_xlabel("Time in fs")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(orig_imag, label="original pulse", color="green")
    axs[2].plot(pred_imag, label="predicted pulse", color="red")
    axs[2].set_ylabel("Imaginary Part of E-Field")
    axs[2].set_xlabel("Time in fs")
    axs[2].grid(True)
    axs[2].legend()
    
    # Plot Intensity difference
    intensity_diff = orig_abs - pred_abs
    axs[3].plot(intensity_diff, color="blue")
    axs[3].set_ylabel("Intensity difference of the original and predicted pulse")
    axs[3].set_xlabel("Time in fs")
    axs[3].grid(True)
    axs[3].legend() 

    plt.tight_layout()
    plt.savefig("./prediction.png")
    plt.close()


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
