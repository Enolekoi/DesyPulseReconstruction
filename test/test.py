'''
test.py Script

Script used for testing
'''
from modules import config
from modules import helper
from modules import loss
from modules import preprocessing
from modules import constants as c

import copy
import torch
import matplotlib.pyplot as plt
import numpy as np

# Define Paths
PathSpec = "./additional/samples/as_gn00.dat"
PathLabel = "./additional/samples/Es.dat"

# Initialize transforms for the spectrograms
shg_reader = helper.ReadSHGmatrix()
shg_transform = helper.ResampleSHGmatrix(    
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
read in label and SHG-matrix
'''
# get label
label = label_reader(PathLabel)
label = label_ambig(label)
# create analytical signal from label
label_size = label.size(0)
label_real = label[:label_size // 2]
label_imag = label[label_size // 2:]
label_analytical = torch.complex(label_real, label_imag)

# read and resample SHG-matrix from file
shg_data_file = shg_reader(PathSpec)

original_shg_not_resampled, original_header,\
original_shg, original_output_time, original_output_wavelength = shg_transform(shg_data_file)

# get the information from the SHG-matrix header
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
temp_freq_axis = helper.frequencyAxisFromHeader(new_header)

# get center frequency
new_center_freq = helper.getCenterOfAxis(temp_freq_axis)

# calculate SHG-Matrix from analytical signal
new_shg = loss.createSHGmat(label_analytical, delta_tau, new_center_freq / 2)
# get the intensity SHG-Matrix
new_shg = torch.abs(new_shg) ** 2
# normalize the SHG-matrix to [0, 1]
new_shg = helper.normalizeSHGmatrix(new_shg)

# calculate angular frequency step between samples
delta_nu = 1 / (N * delta_tau) 
delta_omega = 2 * c.pi * delta_nu

# get delay_axis
new_delay_axis = preprocessing.generateAxis(N=num_delays, resolution=delta_tau, center=0.0)
# get new frequency axis
freq_axis = preprocessing.generateAxis(N=num_wavelength, resolution=delta_omega, center=new_center_freq)

print(f"Min frequency value = {freq_axis.min():.3}")
print(f"Max frequency value = {freq_axis.max():.3}")

# convert to wavelength
new_wavelength_axis, new_shg = preprocessing.intensityMatrixFreq2Wavelength(freq_axis, new_shg)
print(f"Min wavelength value = {new_wavelength_axis.min():.3}")
print(f"Max wavelength value = {new_wavelength_axis.max():.3}")
# get new center_wavelength
new_center_wavelength = helper.getCenterOfAxis(new_wavelength_axis)
# calculate wavelength step size between samples
new_delta_lambda = float(new_wavelength_axis[1] - new_wavelength_axis[0])

print(f"New center wavelength           = {(new_center_wavelength):.3}")
print(f"Original center wavelength      = {(center_wavelength):.3}")
print(f"Center wavelength difference    = {(new_center_wavelength - center_wavelength):.3}")

# create the header for the newly created shg-matrix
new_num_delays = new_shg.size(1)
new_num_wavelength = new_num_delays
new_header = [
        new_num_delays,
        new_num_wavelength,
        delta_tau,
        new_delta_lambda,
        new_center_wavelength
        ]

print(f"Original Header = {original_header}")
print(f"New Header      = {new_header}")

# resample new SHG-matrix
shg_data = [new_shg, new_header]

new_shg_not_resampled, new_header,\
new_shg, new_output_time, new_output_wavelength = shg_transform(shg_data)

# get original SHG-matrix (without 3 identical channels)
original_shg = original_shg[1, :, :]
new_shg = new_shg[1, :, :]

# normalize to [0, 1]
original_shg = helper.normalizeSHGmatrix(original_shg)
new_shg = helper.normalizeSHGmatrix(new_shg)

'''
Plot
'''
fig, axs = plt.subplots(3, figsize=(10,14))
# Simuliertes Spektrogram (original)
ax = axs[0]
cax0 = ax.pcolormesh(
        original_output_time.numpy(),
        original_output_wavelength.numpy(),
        original_shg.numpy().T,
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
                     new_shg.numpy().T,
                     shading='auto'
                     )
ax.set_title('Aus Label erstelltes Spektrogramm')
ax.set_xlabel('Time [fs]')
ax.set_ylabel('Wavelength [nm]')
fig.colorbar(cax1, ax=ax)

# 
ax = axs[2]
cax2 = ax.pcolormesh(new_output_time.numpy(),
                     new_output_wavelength.numpy(),
                     new_shg.numpy().T-original_shg.numpy().T,
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
