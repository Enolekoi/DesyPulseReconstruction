import config
import helper
import visualize as vis

import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
import logging


spectrogram_path = "/mnt/data/desy/frog_simulated/grid_512_v2/tr1000/as_gn00.dat"
label_path = "/mnt/data/desy/frog_simulated/grid_512_v2/tr1000/Es.dat"
filepath = "./pred.png"

# Logger Settings
logging.basicConfig(
        level=logging.INFO,
        style="{",
        format="{asctime} - {name} - {funcName} - {levelname}: {message}",
        datefmt='%d-%m-%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(config.log_filepath),
            logging.StreamHandler()
            ]
)
logger = logging.getLogger(__name__)

###########
## MODEL ##
###########
# Define Model
model = helper.CustomDenseNet(
    num_outputs=2*config.OUTPUT_SIZE
    )
# Load the saved model state
model.load_state_dict(torch.load(config.ModelPath, weights_only=True))
logger.info(f"Model state loaded from {config.ModelPath}")
# Set to evaluation mode
model.eval()

################
## TRANSFORMS ##
################
spec_transform = helper.ResampleSpectrogram(config.OUTPUT_NUM_DELAYS, config.OUTPUT_TIMESTEP, config.OUTPUT_NUM_WAVELENGTH, config.OUTPUT_START_WAVELENGTH, config.OUTPUT_END_WAVELENGTH)

label_reader = helper.ReadLabelFromEs(config.OUTPUT_SIZE)
label_phase_correction = helper.RemoveAbsolutePhaseShift()
label_scaler = helper.ScaleLabel(max_intensity=config.MAX_INTENSITY, max_phase=config.MAX_PHASE)
# label_transform = transforms.Compose([label_reader, label_phase_correction, label_scaler])
label_transform = transforms.Compose([label_reader, label_scaler])
label_unscaler = helper.UnscaleLabel(max_intensity=config.MAX_INTENSITY, max_phase=config.MAX_PHASE)

################
## PREDICTION ##
################

def predict(spectrogram):
    with torch.no_grad():
        output_unscaled = model(spectrogram)
        output = label_unscaler(output_unscaled)
    return output

# load spectrogram
logger.info(f"Loading spectrogram from {spectrogram_path}")
orig_spec, orig_time, orig_wave, spec, out_time, out_wave = spec_transform(spectrogram_path)
spec = torch.tensor(spec, dtype=torch.double)
spec = torch.unsqueeze(spec, dim=0)  # Add batch dimension [1, 512, 512]
spec = torch.unsqueeze(spec, dim=0)  # Add batch dimension [1, 1, 512, 512]
spec = spec.repeat(1, 3, 1, 1)  # Repeat the channel 3 times, changing [1, 1, 512, 512] to [1, 3, 512, 512] 
spec = spec.to(torch.double)

label = label_transform(label_path)
prediction = predict(spec)

prediction_length = len(prediction)

vis.compareTimeDomainComplex(filepath, label, prediction)

for handler in logger.handlers:
    handler.flush()
    handler.close()
