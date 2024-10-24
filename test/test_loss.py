'''
test.py Script

Script used for testing
'''
from modules import config
from modules import helper
from modules import loss

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

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

# read and resample spectrogram from file
spec_data_file = spec_reader(PathSpec)
sim_spec, sim_input_time, sim_input_wavelength, sim_output_spec, sim_output_time, sim_output_freq = spec_wave_transform(spec_data_file)

# create analytical signal from label
label_size = label.size(0)
label_real = label[:label_size // 2]
label_imag = label[label_size // 2:]
label = label.unsqueeze(0)
label_real = label_real.unsqueeze(0)
# print(f"calc size  : {label.size(0)}")
# print(f"label size : {label.size(0)}")

loss_fun = loss.PulseRetrievalLossFunctionHilbertFrog(
        pulse_threshold = 0.01,
        real_weight = 1.0,             
        imag_weight = 1.0,
        intensity_weight = 10.0,
        phase_weight = 5.0,
        frog_error_weight= 1.0
        )

frog_error = loss_fun(label_real, label, sim_output_spec.unsqueeze(0))
# print(f"FROG-Error = {frog_error:.4e}")
