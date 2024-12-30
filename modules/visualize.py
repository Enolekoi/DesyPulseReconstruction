'''
visualize.py Module

Module containing functions used for plotting or visualizing data
'''
#############
## Imports ##
#############
import os
import logging
import matplotlib
import tikzplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from modules import helper
from modules import data
from modules import constants as c

logger = logging.getLogger(__name__)

'''
comparePreproccesSHGMatrix()

Description:
    Plot the raw and preprocessed Spectrogram
Inputs:
    raw_filepath        -> [string] Path to raw SHG-matrix
    preproc_filepath    -> [string] Path to preprocessed SHG-matrix
    save_path           -> [string] Save the figure to this path
'''
def comparePreproccesSHGMatrix(raw_filepath, preproc_filepath, save_path):
    reader = data.ReadSHGmatrix()
    try:
        # get the matrix and header
        raw_shg_data = reader(raw_filepath)
        preproc_shg_data = reader(preproc_filepath)
        
        raw_matrix, raw_header = raw_shg_data
        preproc_matrix, preproc_header = preproc_shg_data
        
        # generate axes from the headers
        raw_delay_axis, raw_wavelength_axis = helper.generateAxes(raw_header)
        preproc_delay_axis, preproc_wavelength_axis = helper.generateAxes(preproc_header)
        
        # create a figure with 2 rows and 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(12,12))

        fig.suptitle(f"Comparison between \n'{raw_filepath}' and \n '{preproc_filepath}'", ha='center')
        # ensure all plots are quadratic
        for ax in axes.flat:
            # ax.set_aspect('equal')
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-15,-15))    # use 1e-15 as exponent for x axis
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9))      # use 1e-9  as exponent for y axis
            ax.set_xlabel('Delay in s')
            ax.set_ylabel('Wavelength in m')
        
        # Plot the raw matrix on a linear scale
        im1 = axes[0, 0].pcolormesh(
                raw_delay_axis,
                raw_wavelength_axis,
                raw_matrix.T,
                shading='auto',
                # norm=LogNorm(vmin=1e-10, vmax=float( resampled_shg_matrix.max())
                )
        axes[0, 0].set_title(f'Raw SHG-matrix (Linear)')
        fig.colorbar(im1, ax=axes[0, 0])

        # Plot the raw matrix on a logarithmic scale
        im2 = axes[0, 1].pcolormesh(
                raw_delay_axis,
                raw_wavelength_axis,
                raw_matrix.T,
                shading='auto',
                norm=LogNorm(vmin=1e-10, vmax=float( raw_matrix.max()))
                )
        axes[0, 1].set_title(f'Raw SHG-matrix (Logarithmic)')
        fig.colorbar(im2, ax=axes[0, 1])

        # Plot the preprocessed matrix on a linear scale 
        im3 = axes[1, 0].pcolormesh(
                preproc_delay_axis,
                preproc_wavelength_axis,
                preproc_matrix.T,
                shading='auto',
                # norm=LogNorm(vmin=1e-10, vmax=float( preproc_matrix.max()))
                )
        axes[1, 0].set_title(f'Preprocessed SHG-matrix (Linear)')
        fig.colorbar(im3, ax=axes[1, 0])

        # Plot the preprocessed matrix on a logarithmic scale 
        im4 = axes[1, 1].pcolormesh(
                preproc_delay_axis,
                preproc_wavelength_axis,
                preproc_matrix.T,
                shading='auto',
                norm=LogNorm(vmin=1e-10, vmax=float( preproc_matrix.max()))
                )
        axes[1, 1].set_title(f'Preprocessed SHG-matrix (Logarithmic)')
        fig.colorbar(im4, ax=axes[1, 1])

        # save the figure
        plt.savefig(save_path)
        plt.close(fig)

    except Exception as e:
        logger.error(f"An error occurred: {e}")

'''
savePlotTrainingLoss()

Description:
    Plot the training, validation and test losses
Inputs:
    loss_values     -> [string] Array containing training loss values
    filepath        -> [string] File to write plot to (without filepath)
'''
def save_plot_training_loss(training_loss, validation_loss, learning_rates, train_size, num_epochs, filepath):
    # get correct filepaths
    tikz_filepath = f"{filepath}.tex"
    png_filepath = f"{filepath}.png"

    num_steps = train_size * num_epochs
    if num_steps != len(training_loss):
        num_steps = len(training_loss)
    # create the x-Axis
    steps = np.arange(num_steps)
    # calculate how many steps are needed for each epoch
    steps_per_epoch = num_steps // num_epochs

    # Create a subplot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16,16))

    #############################
    ## Linear Scale Plot (ax1) ##
    #############################   
    
    # plot training loss over time
    ax1.plot(steps, training_loss, label='Training loss', color='blue')
    # plot the validation_loss for each epoch
    if validation_loss:
        for epoch in range(num_epochs):
            start = epoch * steps_per_epoch
            end = (epoch +1) * steps_per_epoch
            if epoch == 0:
                ax1.hlines(validation_loss[epoch], start, end, label='validation loss', color='red', linestyle='dashed')
            else:
                ax1.hlines(validation_loss[epoch], start, end, color='red', linestyle='dashed')
    
    # Epoch ticks
    epoch_ticks = np.arange(1, num_epochs + 1)

    epoch_labels = [f'Epoch {tick}' for tick in epoch_ticks]
    ax1.set_xticks(epoch_ticks)
    ax1.set_xticklabels(epoch_labels, rotation=45)
    
    # print some information
    ax1.set_xlabel('Number of Steps')
    ax1.set_ylabel('Loss (linear)')
    ax1.grid(True)
    ax1.set_title('Training loss over time (linear)')
    
    # Plot learning rates
    ax1_learning_rate = ax1.twinx()
    ax1_learning_rate.plot(
            steps,
            learning_rates, 
            label='Learning Rate', 
            color='green', 
            )
    ax1_learning_rate.set_ylabel('Learning Rate')

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
    if validation_loss:
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
            steps,
            learning_rates, 
            label='Learning Rate', 
            color='green', 
            )

    ax2_learning_rate.set_ylabel('Learning Rate')

    # Combine legends from both axes
    lines_1, labels_1 = ax2.get_legend_handles_labels()
    lines_2, labels_2 = ax2_learning_rate.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.savefig(png_filepath)
    tikzplotlib.save(tikz_filepath)
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
plotTimeDomainFromPrediction()

Description:
    Plot time domains of a prediction
Inputs:
    filepath    -> path to where the plot is saved
    prediction  -> predicted signal
'''
def plotTimeDomainFromPrediction(filepath, prediction):
    matplotlib.use('TkAgg')  # Or use 'Qt5Agg', 'MacOSX' depending on your system
    if not isinstance(prediction, np.ndarray):
        prediction = prediction.numpy()
    # get the real and imaginary parts
    prediction = prediction.squeeze()

    prediction_length = len(prediction)
    
    half_size = int(prediction_length //2)
    pred_real = prediction[:half_size].squeeze()
    pred_imag = prediction[half_size:].squeeze()
    
    # create the delay axis
    delay_axis = helper.generateAxis(
            N = 256,
            resolution = 1.5*c.femto,
            center = 0.0
            ).numpy()

    # calculate the intensity and phase
    pred_intensity = pred_real*pred_real + pred_imag*pred_imag
    
    pred_phase = np.mod(np.arctan2(pred_imag, pred_real), 2*np.pi)
    pred_phase = np.unwrap(pred_phase)
    
    # calculate the spectral intensity and phase
    pred_complex = pred_real + 1j * pred_imag
    # compute the fft
    pred_fft = np.fft.fft(pred_complex) 
    # calculate the spectral intensity and phase
    fft_intensity = np.sqrt(np.abs(pred_fft))
    fft_phase = np.mod(np.arctan2(pred_fft.imag, pred_fft.real), 2*np.pi)
    fft_phase = np.unwrap(fft_phase)
    # create a frequency axis for plotting
    n = pred_complex.size
    n_half = n // 2
    freq_axis = np.fft.fftfreq(n, d=(1/1.5*c.femto) )

    fft_intensity = fft_intensity[n_half:]
    fft_phase =         fft_phase[n_half:]
    freq_axis =         freq_axis[:n_half]

    fig, axs = plt.subplots(4,1, figsize=(10,16))

    # Plotting the real part
    axs[0].plot(delay_axis, pred_real, label="Retrieved real part", color="red")
    axs[0].set_title("Retrieved real part of the E-Field")
    axs[0].set_xlabel("Delays in [s]")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    axs[0].legend()

    # Plotting the imgainary part
    axs[1].plot(delay_axis,pred_imag, label="Retrieved imaginary part", color="red")
    axs[1].set_title("Retrieved imaginary part of the E-Field")
    axs[1].set_xlabel("Delays in [s]")
    axs[0].set_ylabel("Amplitude")
    axs[1].grid(True)
    axs[1].legend()
    
    # Plotting the Phase
    axs[2].plot(delay_axis,pred_phase, label="Retrieved phase", color="blue")
    axs[2].set_title("Retrieved intensity and phase of the E-Field")
    axs[2].set_ylabel("Phase in [rad]")
    axs[2].set_xlabel("Delays in [s]")
    axs[2].grid(True)
    
    ax_intensity = axs[2].twinx()
    ax_intensity.plot(delay_axis, pred_intensity, label="Retrieved intensity", color="orange")
    ax_intensity.set_ylabel("Intensity")
    # ax_intensity.legend(loc='best')

    # Combine legends from both axes
    lines_1, labels_1 = axs[2].get_legend_handles_labels()
    lines_2, labels_2 = ax_intensity.get_legend_handles_labels()
    axs[2].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    # Plot Intensity difference
    axs[3].plot(freq_axis,fft_phase, label="Retrieved spectral phase", color="blue")
    axs[3].set_title("Retrieved spectral intensity and phase")
    axs[3].set_ylabel("Phase in [rad]")
    axs[3].set_xlabel("Frequency")
    axs[3].grid(True)

    ax_fft_intensity = axs[3].twinx()
    ax_fft_intensity.plot(freq_axis,fft_intensity, label="Retrieved spectral intensity", color="orange")
    ax_fft_intensity.set_ylabel("Intensity")

    lines_3, labels_3 = axs[3].get_legend_handles_labels()
    lines_4, labels_4 = ax_fft_intensity.get_legend_handles_labels()
    axs[3].legend(lines_3 + lines_4, labels_3 + labels_4, loc='upper right')

    # Adjust the spacing between plots
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.5)
    plt.savefig(filepath)
    plt.close()
    logger.info(f"Saved plot of prediction to {filepath}")

'''
compareTimeDomainComplex()

Description:
    Compate Time Domains of label and prediction
Inputs:
    filepath    -> [string] path to where the plot is saved (without fileending)
    label       -> [tensor] label of data
    prediction  -> [tensor] predicted data
'''
def compareTimeDomainComplex(filepath, label, prediction):
    # get correct filepaths
    tikz_filepath = f"{filepath}.tex"
    png_filepath =  f"{filepath}.png"

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
        logger.error(f"Label length:      {len(label)}")
        logger.error(f"Prediction length: {len(prediction)}")
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

    # Plotting the real part
    axs[0].plot(orig_real, label="original pulse", color="green")
    axs[0].plot(pred_real, label="predicted pulse", color="red")
    axs[0].set_title("Comparison of the real part of the E-Field")
    axs[0].set_xlabel("Time in fs")
    axs[0].grid(True)
    axs[0].legend()

    # Plotting the imaginary part
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
    lines_1, labels_1 = axs[2].get_legend_handles_labels()
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
    plt.savefig(png_filepath)
    tikzplotlib.save(tikz_filepath)
    plt.close()
    logger.info(f"Saved comparison of random prediction and label to {filepath}")

'''
plotSpectrogram()
plotting Spectrogram
'''
def plotSpectrogram(path):
    # reader
    reader = data.ReadSHGmatrix()
    shg_data = reader(path)
    shg_matrix, header = shg_data

    delay_axis, wavelength_axis = helper.generateAxes(header)

    # Create a figure
    plt.figure(figsize=(10, 18))

    # FIGURE 1
    # fix the axes to a specific exponent representation
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-15,-15))    # use 1e-15 as exponent for x axis
    plt.ticklabel_format(axis="y", style="sci", scilimits=(-9,-9))      # use 1e-9  as exponent for y axis
    # Plot the SHG-matrix
    plt.pcolormesh(
        delay_axis.numpy(),
        wavelength_axis.numpy(),
        shg_matrix.numpy().T,
        shading='auto',
        norm=LogNorm(vmin=1e-10, vmax=float( shg_matrix.max() ))
        )

    # Add labels and title
    # fig.colorbar(c1, label='Intensity')
    plt.ylabel("Wavelength [m]")
    plt.xlabel("Time [s]")
    plt.title("SHG-Matrix")
    
    plt.colorbar(label='Intensity')
    plt.show()
    return
