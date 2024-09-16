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
def save_plot_training_loss(training_loss, validation_loss, learning_rates, train_size, num_epochs, filepath):
    num_steps = train_size * num_epochs
    if num_steps != len(training_loss):
        num_steps = len(training_loss)
    # create the x-Axis
    steps = np.arange(num_steps)
    # calculate how many steps are needed for each epoch
    steps_per_epoch = num_steps // num_epochs

    # Create a subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14,14))

    #############################
    ## Linear Scale Plot (ax1) ##
    #############################   

    # plot training loss over time
    ax1.plot(steps, training_loss, label='Training loss', color='blue')
    # plot the validation_loss for each epoch
    for epoch in range(num_epochs):
        start = epoch * steps_per_epoch
        end = (epoch +1) * steps_per_epoch
        if epoch == 0:
            ax1.hlines(validation_loss[epoch], start, end, label='validation loss', color='red', linestyle='dashed')
        else:
            ax1.hlines(validation_loss[epoch], start, end, color='red', linestyle='dashed')
    
    # Epoch ticks
    epoch_ticks = np.arange(steps_per_epoch, num_steps + 1, steps_per_epoch)
    epoch_labels = [f'Epoch {i+1}' for i in range(num_epochs)]
    ax1.set_xticks(epoch_ticks)
    ax1.set_xticklabels(epoch_labels, rotation=45)
    
    # print some information
    ax1.set_xlabel('Number of Steps')
    ax1.set_ylabel('Loss (linear)')
    ax1.grid(True)
    ax1.set_title('Training loss over time (linear)')
    
    # Plot learning rates
    print(np.shape(learning_rates))
    print(type(learning_rates))
    ax1_learning_rate = ax1.twinx()
    ax1_learning_rate.plot(
            np.arange(steps_per_epoch, num_steps+1, steps_per_epoch), 
            learning_rates, 
            label='Learning Rate', 
            color='green', 
            marker='o'
            )
    # ax1_learning_rate.set_ylabel('Learning Rate')

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_learning_rate.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    ##################################
    ## Logarithmic Scale Plot (ax2) ##
    ##################################

    # plot training loss over time
    ax2.plot(steps, training_loss, label='Training loss', color='blue')
    
    # Set y-axis to log scale
    ax2.set_yscale('log')
    
    # plot the validation_loss for each epoch
    for epoch in range(num_epochs):
        start = epoch * steps_per_epoch
        end = (epoch +1) * steps_per_epoch
        if epoch == 0:
            ax2.hlines(validation_loss[epoch], start, end, label='validation loss', color='red', linestyle='dashed')
        else:
            ax2.hlines(validation_loss[epoch], start, end, color='red', linestyle='dashed')
    
    # Epoch ticks
    ax2.set_xticks(epoch_ticks)
    ax2.set_xticklabels(epoch_labels, rotation=45)
    
    # print some information
    ax2.set_xlabel('Number of Steps')
    ax2.set_ylabel('Loss (logarithmic)')
    ax2.grid(True)
    ax2.set_title('Training loss over time (logarithmic)')
    
    # Plot learning rates
    ax2_learning_rate = ax2.twinx()
    ax2_learning_rate.plot(
            np.arange(steps_per_epoch, num_steps+1, steps_per_epoch), 
            learning_rates, 
            label='Learning Rate', 
            color='green', 
            marker='o'
            )
    ax2_learning_rate.set_ylabel('Learning Rate')

    # Combine legends from both axes
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_learning_rate.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.savefig(filepath)
    plt.close()

    logger.info(f"Plot writen to {filepath}")

'''
compareIntensity()
Compate Time Domains of label and prediction
Inputs:
    filepath    -> path to where the plot is saved
    label       -> label of data
    prediction  -> predicted data
'''
def compareIntensity(filepath, label, prediction):
    # ensure correct datatype
    if not isinstance(label, np.ndarray):
        label = label.cpu().numpy()
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.cpu().numpy()
    label = np.squeeze(label)
    prediction = np.squeeze(prediction)
    
    length_label = len(label)
    if not (length_label == len(prediction)):
        logger.error("Label and prediction don't have the same size")
        return

    fig, axs = plt.subplots(2,1, figsize=(8,14))

    # Plotting the Phase
    axs[0].plot(label, label="original pulse", color="green")
    axs[0].plot(prediction, label="predicted pulse", color="red")
    axs[0].set_title("Comparison of the Intensity of the E-Field")
    axs[0].set_ylabel("Intensity of E-Field")
    axs[0].set_xlabel("Time in fs")
    axs[0].grid(True)
    axs[0].legend()

    # Plot Intensity difference
    intensity_diff = label - prediction
    axs[1].plot(intensity_diff, color="blue")
    axs[1].set_title("Intensity difference of the original and predicted pulse")
    axs[1].set_ylabel("Intensity difference of the original and predicted pulse")
    axs[1].set_xlabel("Time in fs")
    axs[1].grid(True)

    # Adjust the spacing between plots
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved comparison of random prediction and label to {filepath}")

'''
comparePhase()
Compate Time Domains of label and prediction
Inputs:
    filepath    -> path to where the plot is saved
    label       -> label of data
    prediction  -> predicted data
'''
def comparePhase(filepath, label, prediction):
    # ensure correct datatype
    if not isinstance(label, np.ndarray):
        label = label.numpy()
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.numpy()
    label = np.squeeze(label)
    prediction = np.squeeze(prediction)
    
    length_label = len(label)
    if not (length_label == len(prediction)):
        logger.error("Label and prediction don't have the same size")
        return

    label = np.unwrap(label)
    prediction = np.unwrap(prediction)

    fig, axs = plt.subplots(2,1, figsize=(8,14))

    # Plotting the Phase
    axs[0].plot(label, label="original pulse", color="green")
    axs[0].plot(prediction, label="predicted pulse", color="red")
    axs[0].set_title("Comparison of the Phase of the E-Field")
    axs[0].set_ylabel("Intensity of E-Field")
    axs[0].set_xlabel("Time in fs")
    axs[0].grid(True)
    axs[0].legend()

    # Plot Phase difference
    phase_diff = label - prediction
    axs[1].plot(phase_diff, color="blue")
    axs[1].set_title("Phase difference of the original and predicted pulse")
    axs[1].set_ylabel("Phase difference of the original and predicted pulse")
    axs[1].set_xlabel("Time in fs")
    axs[1].grid(True)

    # Adjust the spacing between plots
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved comparison of random prediction and label to {filepath}")


'''
compareTimeDomain()
Compate Time Domains of label and prediction
Inputs:
    filepath    -> path to where the plot is saved
    label       -> label of data
    prediction  -> predicted data
'''
def compareTimeDomain(filepath, label, prediction):
    # ensure correct datatype
    if not isinstance(label, np.ndarray):
        label = label.numpy()
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.numpy()
    label = np.squeeze(label)
    prediction = np.squeeze(prediction)
    
    length_label = len(label)
    if not (length_label == len(prediction)):
        logger.error("Label and prediction don't have the same size")
        return
    half_size = int(length_label //2)

    orig_intensity = label[:half_size]
    orig_phase = label[half_size:]
    pred_intensity = prediction[:half_size]
    pred_phase = prediction[half_size:]
    
    orig_phase = np.unwrap(orig_phase)
    pred_phase = np.unwrap(pred_phase)

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
    axs[2].plot(intensity_diff, color="blue")
    axs[2].set_title("Intensity difference of the original and predicted pulse")
    axs[2].set_ylabel("Intensity difference of the original and predicted pulse")
    axs[2].set_xlabel("Time in fs")
    axs[2].grid(True)

    # Adjust the spacing between plots
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved comparison of random prediction and label to {filepath}")

'''
compareTimeDomainComplex()
Compate Time Domains of label and prediction
Inputs:
    filepath    -> path to where the plot is saved
    label       -> label of data
    prediction  -> predicted data
'''
def compareTimeDomainComplex(filepath, label, prediction):
    # ensure correct datatype
    if not isinstance(label, np.ndarray):
        label = label.numpy()
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.numpy()
    label = np.squeeze(label)
    prediction = np.squeeze(prediction)
    
    length_label = len(label)
    if not (length_label == len(prediction)):
        logger.error("Label and prediction don't have the same size")
        return
    half_size = int(length_label //2)
    orig_real = label[:half_size]
    orig_imag = label[half_size:]
    pred_real = prediction[:half_size]
    pred_imag = prediction[half_size:]

    orig_intensity = orig_real*orig_real + orig_imag*orig_imag
    pred_intensity = pred_real*pred_real + pred_imag*pred_imag
    orig_phase = np.mod(np.arctan2(orig_imag, orig_real), 2*np.pi)
    pred_phase = np.mod(np.arctan2(pred_imag, pred_real), 2*np.pi)

    orig_phase = np.unwrap(orig_phase)
    pred_phase = np.unwrap(pred_phase)

    fig, axs = plt.subplots(4,1, figsize=(10,16))

    # Plotting the Intensity
    axs[0].plot(orig_real, label="original pulse", color="green")
    axs[0].plot(pred_real, label="predicted pulse", color="red")
    axs[0].set_title("Comparison of the real part of the E-Field")
    axs[0].set_xlabel("Time in fs")
    axs[0].grid(True)
    axs[0].legend()

    # Plotting the Phase
    axs[1].plot(orig_imag, label="original pulse", color="green")
    axs[1].plot(pred_imag, label="predicted pulse", color="red")
    axs[1].set_title("Comparison of the imaginary part of the E-Field")
    axs[1].set_xlabel("Time in fs")
    axs[1].grid(True)
    axs[1].legend()
    
    # Plotting the Phase
    axs[2].plot(orig_phase, label="original phase", color="green")
    axs[2].plot(pred_phase, label="predicted phase", color="blue")
    axs[2].set_title("Comparison of phase and intensity of the E-Field")
    axs[2].set_ylabel("Phase (rad)")
    axs[2].set_xlabel("Time in fs")
    axs[2].grid(True)
    
    ax_intensity = axs[2].twinx()
    ax_intensity.plot(orig_intensity, label="original intensity", color="red")
    ax_intensity.plot(pred_intensity, label="predicted intensity", color="orange")
    ax_intensity.set_ylabel("Intensity")
    # ax_intensity.legend(loc='best')

    # Combine legends from both axes
    lines_1, labels_1 = ax[2].get_legend_handles_labels()
    lines_2, labels_2 = ax_intensity.get_legend_handles_labels()
    axs[2].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

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

'''
plotSpectrogram()
plotting Spectrogram
'''
def plotSpectrogram(spectrogram):
    return
