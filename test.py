'''
test.py Script

Script used for testing
'''
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
Ts = 1.5e-15
Center = 337.927e-9
Step = 0.88e-9
N = 256


# Initialize Resampler
spec_reader = helper.ReadSpectrogram()
spec_wave_transform = helper.ResampleSpectrogram(
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_FREQUENCIES,
    config.OUTPUT_START_FREQUENCY,
    config.OUTPUT_END_FREQUENCY,
    type='wavelength'
    )
spec_freq_transform = helper.ResampleSpectrogram(    
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_FREQUENCIES,
    config.OUTPUT_START_FREQUENCY,
    config.OUTPUT_END_FREQUENCY,
    type='frequency'
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

# calculate input wavelength axis
Start = Center - (N/2)*Step      # calculate the first element of the wavelength input axis
End = Center + (N/2)*Step       # calculate the last element of the wavelength input axis
input_wave = torch.linspace(Start, End, N)    # create array that corresponds tot the input wavelength axis
input_freq = (2* torch.pi * SPEED_OF_LIGHT) / input_wave # convert wavelenght [nm] to frequency [Hz]

# centerIdx = len(input_freq) // 2
# wCenter = input_freq[centerIdx - 1]
wCenter = (2* torch.pi * SPEED_OF_LIGHT) / Center # convert wavelenght [nm] to frequency [Hz]

tStart = -int(N / 2) * Ts    # calculate time at which the input time axis starts
tEnd = tStart + (Ts * N) - Ts    # calculate the last element of the input time axis
input_time = torch.linspace(tStart, tEnd, N)   # create array that corresponds to the input time axis
        
# calculate SHG Matrix from analytical signal
spec = loss.createSHGmat(label_analytical, Ts, wCenter)
spec = torch.abs(spec)**2
# resample SHG Matrix
spec_data = [spec, input_time, input_freq]
sim, input_time, input_freq, spec, spec_output_time, spec_output_freq = spec_freq_transform(spec_data)
# print(f"owStart = {spec_output_freq.min():.4e}")
# print(f"owEnd = {spec_output_freq.max():.4e}")
# print(f"otStart = {spec_output_time.min():.4e}")
# print(f"otEnd = {spec_output_time.max():.4e}")

# read and resample spectrogram from file
spec_data_file = spec_reader(PathSpec)
sim_spec, sim_input_time, sim_input_wavelength, sim_output_spec, sim_output_time, sim_output_freq = spec_wave_transform(spec_data_file)

# get original spectrogram (without 3 identical channels)
original_spectrogram = sim_output_spec
print(f"og_spec min: {torch.min(original_spectrogram)}")
print(f"og_spec max: {torch.max(original_spectrogram)}")

loss.calcFrogError(original_spectrogram, spec)
# Plot
fig, axs = plt.subplots(2, figsize=(10,8))
# Simuliertes Spektrogram (original)
ax = axs[0]
cax1 = ax.pcolormesh(original_spectrogram.numpy(), shading='auto')
ax.set_title('Simuliertes Spektrum (original)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Wavelength [nm]')
fig.colorbar(cax1, ax=ax)

# Simuliertes Spektrogram (rekonstruiert)
ax = axs[1]
cax2 = ax.pcolormesh(spec.numpy(), shading='auto')
ax.set_title('Simuliertes Spektrum (rekonstruiert)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Frequency [Hz]')
fig.colorbar(cax2, ax=ax)

# Layout anpassen plt.tight_layout()

# Zeige die Plots an
plt.show()
