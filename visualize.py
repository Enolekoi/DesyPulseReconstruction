import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

# TODO:
'''
save_plot_training_loss()
Plot the training loss ( Currently UNUSED )
Inputs:
    loss_values     -> Array containing training loss values
    filepath        -> File to write plot to
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
compareTimeDomain()
Compate Time Domains of label and prediction
Inputs:
    filepath    -> path to where the plot is saved
    label       -> label of data
    prediction  -> predicted data
'''
def compareTimeDomain(filepath, label, prediction):
    
    orig_intensity = label[:256].numpy()
    orig_phase = label[256:].numpy()
    pred_intensity = prediction[:256].numpy()
    pred_phase = prediction[256:].numpy()
    
    fig, axs = plt.subplots(3,1, figsize=(8,14))

    # Plotting the Intensity
    axs[0].plot(orig_intensity, label="original pulse", color="green")
    axs[0].plot(pred_intensity, label="predicted pulse", color="red")
    axs[0].set_title("Comparison of the intensities of the E-Field")
    axs[0].set_ylabel("Intensity of E-Field")
    axs[0].set_xlabel("Time in fs")
    axs[0].grid(True)
    axs[0].legend()

    # Plotting the Phase
    axs[1].plot(orig_phase, label="original pulse", color="green")
    axs[1].plot(pred_phase, label="predicted pulse", color="red")
    axs[1].set_title("Comparison of the phase of the E-Field")
    axs[1].set_ylabel("Phase of E-Field")
    axs[1].set_xlabel("Time in fs")
    axs[1].grid(True)
    axs[1].legend()

    # Plot Intensity difference
    intensity_diff = orig_intensity - pred_intensity
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
