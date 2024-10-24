'''
test.py Script

Script used for testing
'''
from modules import config
from modules import helper
from modules import loss
from modules import preprocessing

import copy
import torch
import matplotlib.pyplot as plt
import numpy as np

# Define Paths
PathSpec = "./additional/samples/as_gn00.dat"
PathLabel = "./additional/samples/Es.dat"

# Constants
c0 = 299792458
c2p = 2 * torch.pi *c0

# Initialize transforms for the spectrograms
spec_reader = helper.ReadSpectrogram()
spec_transform = helper.ResampleSpectrogram(    
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_WAVELENGTH,
    config.OUTPUT_START_WAVELENGTH,
    config.OUTPUT_END_WAVELENGTH,
    )

# Initialize transforms for the labels
label_reader = helper.ReadLabelFromEs(config.OUTPUT_SIZE)
label_ambig = helper.RemoveAmbiguitiesFromLabel(config.OUTPUT_SIZE)

'''
read in label and spectrogram
'''
# get label
label = label_reader(PathLabel)
label = label_ambig(label)
# create analytical signal from label
label_size = label.size(0)
label_real = label[:label_size // 2]
label_imag = label[label_size // 2:]
label_analytical = torch.complex(label_real, label_imag)

# read and resample spectrogram from file
spec_data_file = spec_reader(PathSpec)

original_spectrogram_not_resampled, original_header,\
original_spectrogram, original_output_time, original_output_wavelength = spec_transform(spec_data_file)

# get the information from the spectrogram header
num_delays          = original_header[0]
num_wavelength      = original_header[1]
delta_tau           = original_header[2]
delta_lambda        = original_header[3]
center_wavelength   = original_header[4]

assert num_delays == num_wavelength
N = num_wavelength

new_header = original_header

'''
create the new spectrogram from header and label (time domain signal)
'''
# get frequency axis from header
temp_freq_axis = helper.frequency_axis_from_header(new_header)

# get center frequency
new_center_freq = helper.getCenterOfAxis(temp_freq_axis)

# get delay_axis
new_delay_axis = preprocessing.generateAxis(N=num_delays, resolution=delta_tau, center=0.0)

# calculate SHG Matrix from analytical signal
new_spectrogram = loss.createSHGmat(label_analytical, delta_tau, new_center_freq // 2)
# get the intensity SHG Matrix
new_spectrogram = torch.abs(new_spectrogram) ** 2
# normalize the spectrogram to [0, 1]
new_spectrogram = (new_spectrogram - new_spectrogram.min()) / (new_spectrogram.max()-new_spectrogram.min())

delta_nu = 1 / (N * delta_tau) 
delta_omega = 2 * torch.pi * delta_nu

# construct new frequency axis
freq_axis = preprocessing.generateAxis(N=num_wavelength, resolution=delta_nu, center=new_center_freq)

# convert to wavelength
wavelength_axis, new_spectrogram = preprocessing.intensityMatrixFreq2Wavelength(freq_axis, new_spectrogram)
new_center_wavelength = helper.getCenterOfAxis(wavelength_axis)
new_delta_lambda = wavelength_axis[1] - wavelength_axis[0]
print(f"new_center_wavelength       = {new_center_wavelength:.3e}")
print(f"original_center_wavelength  = {center_wavelength:.3e}")
print(f"difference                  = {(new_center_wavelength - center_wavelength):.3e}")

new_header = [
        256,
        256,
        delta_tau,
        delta_lambda,
        new_center_wavelength
        ]
# resample new spectrogram
spec_data = [new_spectrogram, new_header]

new_spectrogram_not_resampled, new_header,\
new_spectrogram, new_output_time, new_output_wavelength = spec_transform(spec_data)

# get original spectrogram (without 3 identical channels)
new_spectrogram = new_spectrogram[1, :, :]
original_spectrogram = original_spectrogram[1, :, :]

# normalize to [0, 1]
original_spectrogram = (original_spectrogram - original_spectrogram.min()) / (original_spectrogram.max() - original_spectrogram.min())
new_spectrogram = (new_spectrogram - new_spectrogram.min()) / (new_spectrogram.max()-new_spectrogram.min())

'''
Plot
'''
fig, axs = plt.subplots(3, figsize=(10,14))
# Simuliertes Spektrogram (original)
ax = axs[0]
cax0 = ax.pcolormesh(
        original_output_time.numpy(),
        original_output_wavelength.numpy(),
        original_spectrogram.numpy().T,
        shading='auto'
        )
ax.set_title('Originales Spektrum')
ax.set_xlabel('Time [fs]')
ax.set_ylabel('Wavelength [nm]')
fig.colorbar(cax0, ax=ax)

# Simuliertes Spektrogram (rekonstruiert)
ax = axs[1]
cax1 = ax.pcolormesh(new_output_time.numpy(),
                     new_output_wavelength.numpy(),
                     new_spectrogram.numpy().T,
                     shading='auto'
                     )
ax.set_title('Aus Label erstelltes Spektrogramm')
ax.set_xlabel('Time [fs]')
ax.set_ylabel('Wavelength [nm]')
fig.colorbar(cax1, ax=ax)

# Differenz der Spektrogramme
ax = axs[2]
cax2 = ax.pcolormesh(new_output_time.numpy(),
                     new_output_wavelength.numpy(),
                     new_spectrogram.numpy().T-original_spectrogram.numpy().T,
                     shading='auto'
                     )
ax.set_title('Differenz')
ax.set_xlabel('Time [fs]')
ax.set_ylabel('Wavelength [nm]')
fig.colorbar(cax2, ax=ax)
# Layout anpassen 
plt.tight_layout()

# Zeige die Plots an
plt.savefig("comparison_spectrograms.png")
plt.show()
# plt.close()
