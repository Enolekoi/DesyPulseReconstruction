'''
test.py Script

Script used for testing
'''
from modules import loss
from modules import config
from modules import data
from modules import helper

import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch.fft as trafo

PATH = "./additional/samples/Es.dat"

label_reader = data.ReadLabelFromEs(config.OUTPUT_SIZE)
label_corr = data.RemoveAmbiguitiesFromLabel(config.OUTPUT_SIZE)

label = label_reader(PATH)
label = label_corr(label)
real_label = label[:config.OUTPUT_SIZE]
imag_label = label[config.OUTPUT_SIZE:]

label_analytical = torch.complex(real_label, imag_label)
label_phase = helper.unwrap_phase(real_label, imag_label).numpy()

analytical_signal = loss.hilbert(real_label, plot=True)
analytical_signal = torch.cat((analytical_signal.real, analytical_signal.imag))
analytical_signal = label_corr(analytical_signal)
half_lenght = len(analytical_signal) // 2
analytical_signal = torch.complex(analytical_signal[:half_lenght], analytical_signal[half_lenght:])
# pred_phase = np.flip(helper.unwrap_phase(analytical_signal.real, analytical_signal.imag).numpy())
pred_phase = helper.unwrap_phase(analytical_signal.real, analytical_signal.imag).numpy()

print(f"Sum Label Phase      = {np.sum(label_phase)}")
print(f"Sum Prediction Phase = {np.sum(pred_phase)}")
# time axis
delta_t = 1.5e-15   # 1.5 fs
N = len(real_label)
time_axis = torch.linspace(-N//2 * delta_t, (N//2 - 1) * delta_t, N)

# frequency axis
freq_axis = trafo.fftfreq(N, d=delta_t)
freq_axis = trafo.fftshift(freq_axis)
angular_freq_axis = 2*torch.pi * freq_axis

plt.figure()

plt.subplot(3,1,1)
plt.plot(time_axis, abs(analytical_signal), label='Analytical signal after Hilbert transform', color='g')
plt.plot(time_axis, abs(label_analytical), label='Analytical label signal after Hilbert transform', color='b', linestyle='--')
plt.title('Analytical signal after Hilbert transform')
plt.xlabel('Frequency')
plt.ylabel('Values')
plt.grid(True)
plt.legend()
ax_fft_intensity = plt.twinx()
ax_fft_intensity.plot(time_axis, pred_phase, label="Prediction Phase", color="orange")
ax_fft_intensity.plot(time_axis, label_phase, label="Label Phase", color="red")
ax_fft_intensity.set_ylabel("Phase [rad]")
 
difference_real = np.abs(pred_phase-label_phase)
plt.subplot(3,1,2)
plt.plot(time_axis, difference_real, label='phase difference between the signals', color='g')
plt.title('Comparing Analytical signal')
plt.xlabel('Frequency')
plt.ylabel('Phase')
plt.grid(True)
plt.legend()

difference_intensity = np.abs(np.sqrt(np.abs(analytical_signal))-np.sqrt(np.abs(label_analytical)) )
plt.subplot(3,1,3)
plt.plot(time_axis, difference_intensity, label='difference between the signals', color='g')
plt.title('Comparing Analytical signals')
plt.xlabel('Frequency')
plt.ylabel('Intensity')
plt.grid(True)
plt.legend()

plt.show()
