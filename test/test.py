'''
test.py Script

Script used for testing
'''
import sys
sys.path.append('./modules/')
import copy
import torch
import config
import helper
import loss
import preprocessing
import matplotlib.pyplot as plt
import numpy as np

# Define Paths
PathSpec = "./additional/samples/as_gn00.dat"
PathLabel = "./additional/samples/Es.dat"

# Constants
c0 = 299792458

# Initialize Resampler
spec_reader = helper.ReadSpectrogram()

spec_transform = helper.ResampleSpectrogram(    
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_FREQUENCIES,
    config.OUTPUT_START_FREQUENCY,
    config.OUTPUT_END_FREQUENCY,
    )

# Initialize Label reader
label_reader = helper.ReadLabelFromEs(config.OUTPUT_SIZE)
label_ambig = helper.RemoveAmbiguitiesFromLabel(config.OUTPUT_SIZE)

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
sim_spec, header, sim_output_spec, sim_output_time, sim_output_freq = spec_transform(spec_data_file)

# get frequency axis from header
freq_axis = helper.frequency_axis_from_header(header)

# get center frequency
num_wavelength = header[1]   # number of wavelength samples
center_index = num_wavelength // 2
prediction_center_freq = freq_axis[center_index]

# get \tau and delay_axis
num_delays = header[2]   # time step per delay [s] (\tau)
prediction_time_step = header[2]   # time step per delay [s] (\tau)
delay_axis = preprocessing.generateAxis(N=num_delays, resolution=prediction_time_step, center=0.0)

# calculate SHG Matrix from analytical signal
spec = loss.createSHGmat(label_analytical, prediction_time_step, prediction_center_freq)
spec = torch.abs(spec)**2

# calculate \tau_{p}^{FWHM} and \nu_{p}^{FWHM}
# \tau_{p}^{FWHM}
mean_delay_profile = torch.mean(spec, dim=1)
fwhm_delay = preprocessing.calcFWHM(mean_delay_profile, delay_axis)

# \nu_{p}^{FWHM}
mean_freq_profile = torch.mean(spec, dim=0)
fwhm_freq = preprocessing.calcFWHM(mean_freq_profile, freq_axis)

# calculate M (Trebino 215)
# using freq
M = torch.sqrt(fwhm_delay * fwhm_freq * N)
# using wavelength
# get center wavelength 
center_wavelength = header[4]
M = torch.sqrt(fwhm_delay * fwhm_wavelength * N * c0 / center_wavelength**2)

# get delta frequency (delta_nu)
delta_tau = fwhm_delay / M
delta_nu = M/N * (1 / fwhm_delay)

# construct new frequency axis
freq_axis = preprocessing.generateAxis(N=num_wavelength, resolution=delta_nu, center=prediction_center_freq)

# convert to wavelength
preprocessing.
# resample

# resample SHG Matrix
spec_data = [spec, header]
sim, prediction_header, spec, spec_output_time, spec_output_freq = spec_transform(spec_data)

# get original spectrogram (without 3 identical channels)
original_spectrogram = sim_output_spec
# frog_error = loss.calcFrogError(original_spectrogram, spec)

# Plot
fig, axs = plt.subplots(3, figsize=(10,14))
# Simuliertes Spektrogram (original)
ax = axs[0]
cax0 = ax.pcolormesh(sim_output_time.numpy(), sim_output_freq.numpy(), original_spectrogram.numpy().T, shading='auto')
# cax0 = ax.pcolormesh(sim_spec.squeeze().numpy().T, shading='auto')
ax.set_title('Originales Spektrum')
ax.set_xlabel('Time [fs]')
ax.set_ylabel('Frequency [1/s]')
fig.colorbar(cax0, ax=ax)

# Simuliertes Spektrogram (rekonstruiert)
ax = axs[1]
cax1 = ax.pcolormesh(spec_output_time.numpy(), spec_output_freq.numpy(), spec.numpy().T, shading='auto')
ax.set_title('Aus Label erstelltes Spektrogramm')
ax.set_xlabel('Time [fs]')
ax.set_ylabel('Frequency [1/s]')
fig.colorbar(cax1, ax=ax)

# Differenz Spektrogram (rekonstruiert)
ax = axs[2]
cax2 = ax.pcolormesh(spec_output_time.numpy(), spec_output_freq.numpy(), spec.numpy().T-original_spectrogram.numpy().T, shading='auto')
ax.set_title('Differenz')
ax.set_xlabel('Time [fs]')
ax.set_ylabel('Frequency [1/s]')
fig.colorbar(cax2, ax=ax)
# Layout anpassen plt.tight_layout()

# Zeige die Plots an
plt.savefig("comparison_spectrograms.png")
plt.show()
# plt.close()
