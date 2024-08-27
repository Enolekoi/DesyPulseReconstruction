import config
import helper

import torch
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
import logging


spectrogram_path = "./models/trained_model_1.pth"
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
model.load_state_dict(torch.load(config.ModelPath))
logger.info(f"Model state loaded from {config.ModelPath}")
# Set to evaluation mode
model.eval()

################
## TRANSFORMS ##
################
spec_transform = helper.ResampleSpectrogram(config.OUTPUT_NUM_DELAYS, config.OUTPUT_TIMESTEP, config.OUTPUT_NUM_WAVELENGTH, config.OUTPUT_START_WAVELENGTH, config.OUTPUT_END_WAVELENGTH)
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
logger.info("Loading spectrogram from {spectrogram_path}")
spec = spec_transform(spectrogram_path)
prediction = predict(spec)

prediction_length = len(prediction)

half_size = int(prediction_length //2)
intensity = prediction[:half_size]  # First half -> intensity
phase = prediction[half_size:]      # Second half -> phase

fig, ax1 = plt.subplots()
ax1.plot(intensity, color='blue', label="Intensity of E-Field")

ax1.set_xlabel('t in fs')
ax1.set_ylabel('Intensity of E-Field')

ax2 = ax1.twinx()
ax2.plot(phase, color='green', label="Phase of E-Field")
ax2.set_ylabel('Phase of E-Field')

plt.title("Prediction of Time Domain")
plt.savefig(filepath)
plt.close()
# plt.show()

for handler in logger.handlers:
    handler.flush()
    handler.close()
