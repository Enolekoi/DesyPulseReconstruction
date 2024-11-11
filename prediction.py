from modules import config
from modules import data
from modules import helper
from modules import loss
from modules import models
from modules import visualize as vis

import torch
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
import logging



shg_matrix_path = "./additional/samples/as_gn00.dat"
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
##########
## CUDA ##
##########
# If cuda is is available use it instead of the cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device used (cuda/cpu): {device}")
if device == 'cuda':
    torch.cuda.empty_cache()

###########
## MODEL ##
###########
# Define Model
model = models.CustomDenseNetReconstruction(
    num_outputs=config.OUTPUT_SIZE
    )
# Load the saved model state
model.load_state_dict(
    torch.load(
        config.ModelPath,
        weights_only=True,
        map_location=torch.device(device))
    )
logger.info(f"Model state loaded from {config.ModelPath}")
# Set to evaluation mode
model.eval()

################
## TRANSFORMS ##
################
shg_read = data.ReadSHGmatrix()
shg_resampled = data.ResampleSHGmatrix(
    config.OUTPUT_NUM_DELAYS,
    config.OUTPUT_TIMESTEP,
    config.OUTPUT_NUM_WAVELENGTH,
    config.OUTPUT_START_WAVELENGTH,
    config.OUTPUT_END_WAVELENGTH
    )

label_scaler = data.Scaler(
    number_elements=config.OUTPUT_SIZE,
    max_value = 1
    )
label_ambig = data.RemoveAmbiguitiesFromLabel(
        number_elements=config.OUTPUT_SIZE
        )

################
## PREDICTION ##
################

def predict(spectrogram):
    with torch.no_grad():
        real_part = model(spectrogram).squeeze()
        analytical_signal = loss.hilbert(real_part, plot=True)
        output = torch.cat((analytical_signal.real, analytical_signal.imag)).to(device)
        output = label_ambig(output)
    return output

# load SHG-Matrix and convert it to right shape [1, 3, 512, 512]
logger.info(f"Loading SHG-Matrix from {shg_matrix_path}")
shg_data = shg_read(shg_matrix_path)
shg_matrix_original, header, shg_matrix, delay_axis, wavelength_axis = shg_resampled(shg_data)
shg_matrix = torch.unsqueeze(shg_matrix, dim=0)  # Add batch dimension [1, 512, 512]
shg_matrix = shg_matrix.to(torch.float)

prediction = predict(shg_matrix)

vis.plotTimeDomainFromPrediction(filepath, prediction)

for handler in logger.handlers:
    handler.flush()
    handler.close()
