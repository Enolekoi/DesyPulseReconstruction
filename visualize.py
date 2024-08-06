import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

'''
Plot Training Loss
'''
def save_plot_training_loss(loss_values, filepath):

    # plot training loss over time
    plt.plot(loss_values)
    plt.xlabel('Number of Steps')
    plt.ylabel('Loss (logarithmic)')
    plt.yscale('log')
    plt.title('Training loss over time')
    plt.savefig(filepath)
    plt.close()

    logger.info(f"Plot writen to {filepath}")

'''
Plot Training Loss
'''
def compareTimeDomain(filepath, label, prediction):

    orig_real = label[:256]
    orig_imag = label[256:]
    pred_real = prediction[:256]
    pred_imag = prediction[256:]
    
    orig_abs = np.abs(orig_real + 1j* orig_imag) 
    pred_abs = np.abs(pred_real + 1j* pred_imag)

    fig, axs = plt.subplots(4,1, figsize=(8,14))
    # Plotting the Intensity
    axs[0].plot(orig_abs, label="original pulse", color="green")
    axs[0].plot(pred_abs, label="predicted pulse", color="red")
    axs[0].set_title("Comparison of the intensities of the E-Field")
    axs[0].set_ylabel("Intensity of E-Field")
    axs[0].set_xlabel("Time in fs")
    axs[0].grid(True)
    axs[0].legend()

    # Plotting the Real Part
    axs[1].plot(orig_real, label="original pulse", color="green")
    axs[1].plot(pred_real, label="predicted pulse", color="red")
    axs[1].set_title("Comparison of the real parts of the E-Field")
    axs[1].set_ylabel("real part of E-Field")
    axs[1].set_xlabel("Time in fs")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(orig_imag, label="original pulse", color="green")
    axs[2].plot(pred_imag, label="predicted pulse", color="red")
    axs[2].set_title("Comparison of the imaginary parts of the E-Field")
    axs[2].set_ylabel("imaginary part of E-Field")
    axs[2].set_xlabel("Time in fs")
    axs[2].grid(True)
    axs[2].legend()
    
    # Plot Intensity difference
    intensity_diff = orig_abs - pred_abs
    axs[3].plot(intensity_diff, color="blue")
    axs[3].set_title("Intensity difference of the original and predicted pulse")
    axs[3].set_ylabel("Intensity difference of the original and predicted pulse")
    axs[3].set_xlabel("Time in fs")
    axs[3].grid(True)

    # Adjust the spacing between plots
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved comparison of random prediction and label to {filepath}")
