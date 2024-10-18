'''
test.py Script

Script used for testing
'''
import sys
sys.path.append('./modules/')
import loss
import config
import helper

import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch.fft as trafo

PATH = "./additional/samples/Es.dat"

label_reader = helper.ReadLabelFromEs(config.OUTPUT_SIZE)
label_corr = helper.RemoveAmbiguitiesFromLabel(config.OUTPUT_SIZE)

label = label_reader(PATH)
label = label_corr(label)
real_label = label[:config.OUTPUT_SIZE]
imag_label = label[config.OUTPUT_SIZE:]

label_analytical = torch.complex(real_label, imag_label)

analytical_signal = loss.hilbert(real_label, plot=True)

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
 
difference_real = torch.abs(real_label-analytical_signal.real)
plt.subplot(3,1,2)
plt.plot(time_axis, difference_real, label='difference between real signals', color='g')
plt.title('Comparing Analytical signal')
plt.xlabel('Frequency')
plt.ylabel('Values')
plt.grid(True)
plt.legend()

difference_imag = torch.abs(imag_label-analytical_signal.imag)
plt.subplot(3,1,3)
plt.plot(time_axis, difference_imag, label='difference between imaginary signals', color='g')
plt.title('Comparing Analytical signal')
plt.xlabel('Frequency')
plt.ylabel('Values')
plt.grid(True)
plt.legend()

plt.show()


# label_analytical = scipy.signal.hilbert(real_label.numpy())
# hilbert_analytical = loss.hilbert(real_label)

# imag_label = label_analytical.imag
# abs_label = torch.abs(label_analytical)
# # abs_label = torch.abs(torch.from_numpy(label_analytical))
# abs_hilbert = torch.abs(hilbert_analytical)
# abs_label_np = abs_label.numpy()
# abs_hilbert_np = abs_hilbert.numpy()

# abs_difference = abs_hilbert - abs_label
# abs_difference_np = abs_difference.numpy()

# imag_hilbert = hilbert_analytical.imag
# imag_difference = imag_hilbert - imag_label
# imag_difference_np = imag_difference.numpy()

# real_label_np = real_label.numpy()
# imag_label_np = imag_label.numpy()
# imag_label_np = imag_label
# imag_hilbert_np = imag_hilbert.numpy()

# plt.figure()

# # Plot 1
# plt.subplot(5, 1, 1)
# plt.plot(abs_label_np, label='Label Analytical Signal', color='g', linestyle='--')
# plt.plot(abs_hilbert_np, label='Hilbert Analytical Signal', color='m')
# plt.title('Absolute Values of Analytical Signals')
# plt.xlabel('Time')
# plt.ylabel('Values')
# plt.grid(True)
# plt.legend()

# # Plot 2
# plt.subplot(5, 1, 2)
# plt.plot(abs_difference_np, label='Absolute Difference', color='b')
# plt.title('Difference in Absolute Values of Analytical Signals')
# plt.xlabel('Time')
# plt.ylabel('Difference')
# plt.grid(True)
# plt.legend()

# # Plot 3
# plt.subplot(5, 1, 3)
# plt.plot(imag_label_np, label='Label Imaginary Part', color='g', linestyle='--')
# plt.plot(imag_hilbert_np, label='Hilbert Imaginary Part', color='m')
# plt.title('Imaginary Parts of Analytical Signals')
# plt.xlabel('Time')
# plt.ylabel('Imaginary Values')
# plt.grid(True)
# plt.legend()

# # Plot 4
# plt.subplot(5, 1, 4)
# plt.plot(imag_label_np, label='Label Imaginary Part', color='g')
# plt.plot(imag_hilbert_np, label='Hilbert Imaginary Part', color='m', linestyle='--')
# plt.title('Imaginary Parts of Analytical Signals')
# plt.xlabel('Time')
# plt.ylabel('Imaginary Values')
# plt.grid(True)
# plt.legend()

# # Plot 5
# plt.subplot(5, 1, 5)
# plt.plot(real_label_np, label='Label Real Part', color='g')
# plt.title('Real Part of Analytical Signals')
# plt.xlabel('Time')
# plt.ylabel('Imaginary Values')
# plt.grid(True)
# plt.legend()


# # plt.tight_layout()
# plt.show()
