'''
test.py Script

Script used for testing
'''
import copy
import torch
import config
import helper
import loss
import matplotlib.pyplot as plt
import numpy as np

# Define Paths
PathSpec = "./additional/samples/as_gn00.dat"
PathLabel = "./additional/samples/Es.dat"

# Constants
SPEED_OF_LIGHT = 299792458

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

prediction_header = copy.deepcopy(header)
prediction_header.correct_freq_axis()
# calculate SHG Matrix from analytical signal
spec = loss.createSHGmat(label_analytical, prediction_header.delta_time, prediction_header.center_freq)
spec = torch.abs(spec)**2
# resample SHG Matrix
spec_data = [spec, prediction_header]
sim, prediction_header, spec, spec_output_time, spec_output_freq = spec_transform(spec_data)
# print(f"owStart = {spec_output_freq.min():.4e}")
# print(f"owEnd = {spec_output_freq.max():.4e}")
# print(f"otStart = {spec_output_time.min():.4e}")
# print(f"otEnd = {spec_output_time.max():.4e}")

# get original spectrogram (without 3 identical channels)
original_spectrogram = sim_output_spec
# print(f"original spectrogram weigth:  {torch.argmax(original_spectrogram)}")
# print(f"predicted spectrogram weigth:  {torch.argmax(spec)}")
# print(f"original time index with largest values:       {torch.argmax(torch.sum(original_spectrogram, dim=1))}")
# print(f"predicted time index with largest values:      {torch.argmax(torch.sum(spec, dim=1))}")
freq_index_orig = torch.argmax(torch.sum(original_spectrogram, dim=0))
freq_index_pred = torch.argmax(torch.sum(spec, dim=0))
print(f"\noriginal frequency index with largest values:  {freq_index_orig}")
print(f"frequency at index {freq_index_orig} = {spec_output_freq[freq_index_orig]:.4e}")
print(f"center Frequenfy = {header.center_freq:.4e}")
print(f"\npredicted frequency index with largest values: {freq_index_pred}")
print(f"frequency at index {freq_index_pred} = {spec_output_freq[freq_index_pred]:.4e}")
print(f"center Frequenfy = {prediction_header.center_freq:.4e}")

print(f"\ntotal difference (sum): {torch.sum(original_spectrogram-spec):.4e}")

print(f"original start frequency = {torch.min(header.freq_axis):.4e}")
print(f"original end frequency   = {torch.max(header.freq_axis):.4e}")

print(f"predicted start frequency = {torch.min(prediction_header.freq_axis):.4e}")
print(f"predicted end frequency   = {torch.max(prediction_header.freq_axis):.4e}")
frog_error = loss.calcFrogError(original_spectrogram, spec)

# Plot
fig, axs = plt.subplots(3, figsize=(10,8))
# Simuliertes Spektrogram (original)
ax = axs[0]
# cax1 = ax.pcolormesh(original_spectrogram.numpy().T, shading='auto')
cax0 = ax.pcolormesh(sim_spec.squeeze().numpy().T, shading='auto')
ax.set_title('Simuliertes Spektrum (original)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Wavelength [nm]')
fig.colorbar(cax0, ax=ax)

# Simuliertes Spektrogram (rekonstruiert)
ax = axs[1]
cax1 = ax.pcolormesh(spec.numpy().T, shading='auto')
ax.set_title('Simuliertes Spektrum (rekonstruiert)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Frequency [Hz]')
fig.colorbar(cax1, ax=ax)

# Differenz Spektrogram (rekonstruiert)
ax = axs[2]
cax2 = ax.pcolormesh(spec.numpy().T-original_spectrogram.numpy().T, shading='auto')
ax.set_title('Differenz')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Frequency [Hz]')
fig.colorbar(cax2, ax=ax)
# Layout anpassen plt.tight_layout()

# Zeige die Plots an
plt.show()
