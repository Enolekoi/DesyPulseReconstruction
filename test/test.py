'''
test.py Script

Script used for testing
'''
from modules import config
from modules import data
from modules import helper
from modules import loss

import torch
import matplotlib.pyplot as plt

# Define Paths
PathSpec = "./additional/samples/s1/as_gn00.dat"
PathLabel = "./additional/samples/s1/Es.dat"

# Initialize transforms for the spectrograms
shg_reader = data.ReadSHGmatrix()
shg_transform = data.ResampleSHGmatrix(    
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_WAVELENGTH,
    config.OUTPUT_START_WAVELENGTH,
    config.OUTPUT_END_WAVELENGTH,
    )

# Initialize transforms for the labels
label_reader = data.ReadLabelFromEs(config.OUTPUT_SIZE)
label_ambig = data.RemoveAmbiguitiesFromLabel(config.OUTPUT_SIZE)

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

'''
create the new spectrogram from header and label (time domain signal)
'''
new_shg, new_header = loss.createSHGmatFromAnalytical(
        analytical_signal=label_analytical,
        header=original_header
        )

# resample new SHG-matrix
shg_data = [new_shg, new_header]

new_shg_not_resampled, new_header,\
new_shg, new_output_time, new_output_wavelength = shg_transform(shg_data)

# get original SHG-matrix (without 3 identical channels)
# original_shg = original_shg[1, :, :]
# new_shg = new_shg[1, :, :]

# normalize to [0, 1]
original_shg = helper.normalizeSHGmatrix(original_shg)
new_shg = helper.normalizeSHGmatrix(new_shg)

# calculate the FROG-error
frog_error = loss.calcFrogError(t_ref = original_shg, t_meas = new_shg)
print(f"FROG-Error = {frog_error}")

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
plt.savefig("comparison_shg.png")
plt.show()
# plt.close()
