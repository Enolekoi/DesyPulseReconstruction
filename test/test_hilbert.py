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

# PATH = "./additional/samples/Es.dat"
PATH = "./additional/samples/new/s1/Es.dat"

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

plt.figure(figsize=(10,14))

plt.subplot(3,1,1)
plt.plot(time_axis, abs(analytical_signal), label='Analytisches Signal aus Hilbert-Transformation', color='g')
plt.plot(time_axis, abs(label_analytical), label='Analytisches Zielsignal', color='b', linestyle='--')
plt.title('Analytisches Zielsignal und analytisches Signal aus Hilbert-Transformation')
plt.xlabel('Kreisfrequenz [1/s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
ax_fft_intensity = plt.twinx()
ax_fft_intensity.plot(time_axis, pred_phase, label="Hilbert Phase", color="orange")
ax_fft_intensity.plot(time_axis, label_phase, label="Original Phase", color="red")
ax_fft_intensity.set_ylabel("Phase [rad]")
 
difference_real = np.abs(pred_phase-label_phase)
plt.subplot(3,1,2)
plt.plot(time_axis, difference_real, label='Differenz der Phasen', color='g')
plt.title('Vergleich der Phase der analytischen Signale')
plt.xlabel('Kreisfrequenz [1/s]')
plt.ylabel('Phase [rad]')
plt.grid(True)
plt.legend()

difference_intensity = np.abs(np.sqrt(np.abs(analytical_signal))-np.sqrt(np.abs(label_analytical)) )
plt.subplot(3,1,3)
plt.plot(time_axis, difference_intensity, label='Differenz der Intensitäten', color='g')
plt.title('Vergleich der Intensitäten der analytischen Signale')
plt.xlabel('Kreisfrequenz [1/s]')
plt.ylabel('Intensität')
plt.grid(True)
plt.legend()

plt.savefig("hilber1.png")
plt.show()
