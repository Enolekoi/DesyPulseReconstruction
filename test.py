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
spec_transform = helper.ResampleSpectrogram(
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_FREQUENCIES,
    config.OUTPUT_START_FREQUENCY,
    config.OUTPUT_END_FREQUENCY
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
input_wave = np.linspace(Start, End, N)    # create array that corresponds tot the input wavelength axis
input_freq = (2* torch.pi * SPEED_OF_LIGHT * 1e9) / input_wave # convert wavelenght [nm] to frequency [Hz]
input_freq = input_freq[::-1]   # ensure increasing order of frequency
# centerIdx = len(input_freq) // 2
# wCenter = input_freq[centerIdx - 1]
wCenter = (2* torch.pi * SPEED_OF_LIGHT * 1e9) / Center # convert wavelenght [nm] to frequency [Hz]

tStart = -int(N / 2) * Ts    # calculate time at which the input time axis starts
tEnd = tStart + (Ts * N) - Ts    # calculate the last element of the input time axis
input_time = np.linspace(tStart, tEnd, N)   # create array that corresponds to the input time axis
        
# calculate SHG Matrix from analytical signal
spec = loss.createSHGmat(label_analytical, Ts, wCenter)
# resample SHG Matrix
sim, input_time, input_freq, spec, spec_output_time, spec_output_freq = spec_transform.resampleFreq(spec, input_time, input_freq)
print(f"owStart = {spec_output_freq.min():.4e}")
print(f"owEnd = {spec_output_freq.max():.4e}")
print(f"otStart = {spec_output_time.min():.4e}")
print(f"otEnd = {spec_output_time.max():.4e}")

# read and resample spectrogram from file
sim_spec, sim_input_time, sim_input_wavelength, sim_output_spec, sim_output_time, sim_output_freq = spec_transform(PathSpec)
print(f"min wavelenght: {sim_input_wavelength.min():.4e}")
print(f"max wavelenght: {sim_input_wavelength.max():.4e}")

# Plot
fig, axs = plt.subplots(2, figsize=(10,8))
# Simuliertes Spektrogram (original)
ax = axs[0]
cax1 = ax.pcolormesh(sim_output_spec.numpy(), shading='auto')
ax.set_title('Simuliertes Spektrum (original)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Wavelength [nm]')
fig.colorbar(cax1, ax=ax)

# Simuliertes Spektrogram (rekonstruiert)
ax = axs[1]
cax2 = ax.pcolormesh(np.abs(spec.numpy()), shading='auto')
ax.set_title('Simuliertes Spektrum (rekonstruiert)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Frequency [Hz]')
fig.colorbar(cax2, ax=ax)

# Layout anpassen
plt.tight_layout()

# Zeige die Plots an
plt.show()
