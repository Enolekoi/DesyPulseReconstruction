'''
test.py Script

Script used for testing
'''
import torch
import config
import helper
import matplotlib.pyplot as plt
import numpy as np

PathSimulated = "./additional/samples/as_gn00.dat"
PathExperimental = "./additional/samples/spectrogram_1616.txt"

spec_transform = helper.ResampleSpectrogram(
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_FREQUENCIES,
    config.OUTPUT_START_FREQUENCY,
    config.OUTPUT_END_FREQUENCY
    )


sim_spec, sim_input_time, sim_input_wavelength, sim_output_spec, sim_output_time, sim_output_freq = spec_transform(PathSimulated)

exp_spec, exp_input_time, exp_input_wavelength, exp_output_spec, exp_output_time, exp_output_freq = spec_transform(PathExperimental)
# print(exp_output_spec)
# np.savetxt('exp_output_spec.csv', exp_output_spec, delimiter=',')
print(torch.max(exp_output_spec))
# exp_input_time = exp_input_time[:-1]
# exp_input_wavelength = exp_input_wavelength[:-1]
# exp_output_time = exp_output_time[:-1]
# exp_output_freq = exp_output_freq[:-1]
# exp_spec = exp_spec[:-1, :-1]

fig, axs = plt.subplots(2, 2, figsize=(10,8))

# Simuliertes Spektrogram (original)
ax = axs[0, 0]
cax1 = ax.pcolormesh(sim_spec.numpy(), shading='auto')
ax.set_title('Simuliertes Spektrum (original)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Wavelength [nm]')
fig.colorbar(cax1, ax=ax)

# Simuliertes Spektrogram (resampled)
ax = axs[1, 0]
cax2 = ax.pcolormesh(sim_output_spec.numpy(), shading='auto')
ax.set_title('Simuliertes Spektrum (resampled)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Frequency [Hz]')
fig.colorbar(cax2, ax=ax)

# Experimentelles Spektrogram (original)
ax = axs[0, 1]
cax3 = ax.pcolormesh(exp_spec.numpy(), shading='auto')
ax.set_title('Experimentelles Spektrum (original)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Wavelength [nm]')
fig.colorbar(cax3, ax=ax)

# Experimentelles Spektrogram (resampled)
ax = axs[1, 1]
cax4 = ax.pcolormesh(exp_output_spec.numpy(), shading='auto')
ax.set_title('Experimentelles Spektrum (resampled)')
# ax.set_xlabel('Time [fs]')
# ax.set_ylabel('Frequency [Hz]')
fig.colorbar(cax4, ax=ax)

# Layout anpassen
plt.tight_layout()

# Zeige die Plots an
plt.show()
