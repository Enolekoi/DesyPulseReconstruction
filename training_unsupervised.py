'''
training.py Script

Script containing the training of the pulse reconstruction model
'''
#############
## Imports ##
#############
import random
import logging
import matplotlib 
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from modules import helper
from modules import loss as loss_module
from modules import visualize as vis
from modules import config
from modules import data
from modules import models

'''
Variables and settings
'''
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

# Log some information
logger.info(config.DESCRIPTOR)
logger.info("Unsupervised Learning step")
logger.info(f"Writing into log file: {config.log_filepath}")
logger.info(f"Dataset used: {config.Path}")
logger.info(f"SHG-matrix used: {config.ShgFilename}")
logger.info(f"Size of output tensor: {2*config.OUTPUT_SIZE} elements")
logger.info(f"Batch size: {config.BATCH_SIZE} elements")
logger.info(f"Number of epochs: {config.NUM_EPOCHS}")
logger.info(f"Initial learning rate: {1e-6}")
# logger.info(f"Initial learning rate: {config.LEARNING_RATE}")
logger.info(f"Only Pulses with PBDrms lower than {config.TBDRMS_THRESHOLD} are used!")

# Transforms (Inputs)
# Read the SHG-matrix and their headers
shg_read = data.ReadSHGmatrix()
# Resample the SHG-matrix to the same delay and wavelength axes
shg_resample = data.ResampleSHGmatrix(
    config.OUTPUT_NUM_DELAYS, 
    config.OUTPUT_TIMESTEP, 
    config.OUTPUT_NUM_WAVELENGTH,
    config.OUTPUT_START_WAVELENGTH,
    config.OUTPUT_END_WAVELENGTH,
    )
shg_transform = transforms.Compose([shg_read, shg_resample])

# Transforms (Labels)
# Read the Labels
label_reader = data.ReadLabelFromEs(config.OUTPUT_SIZE)
# Remove the trivial ambiguities from the labels
label_remove_ambiguieties = data.RemoveAmbiguitiesFromLabel(config.OUTPUT_SIZE)
# Scale the Labels to the correct amplitude
scaler = data.Scaler(
    number_elements=config.OUTPUT_SIZE, 
    max_value=config.MAX_VALUE
    )
# scale to [0, 1] 
label_scaler = scaler.scale
# scale to original label size
label_unscaler = scaler.unscale
label_transform = transforms.Compose([label_reader, label_remove_ambiguieties, label_scaler])

# If cuda is is available use it instead of the cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device used (cuda/cpu): {device}")
if device == 'cuda':
    torch.cuda.empty_cache()

'''
Load Model
'''
logger.info("Loading Model...")
# Load custom DenseNet
model = models.CustomDenseNetReconstruction(
    num_outputs=config.OUTPUT_SIZE
    )
# Define location of pretrained weights
model.load_state_dict(torch.load('./logs/training_015/model.pth', weights_only=True))

# set the model to float, send it to the selected device and put it in evaluation mode
model.float().to(device).eval()

# Freeze the layers of the densenet
for param in model.densenet.parameters():
    param.requires_grad = False

# Only allow gradients on the last layers after the densenet
for param in model.fc1.parameters():
    param.requires_grad = True
for param in model.fc2.parameters():
    param.requires_grad = True

logger.info("Freezing early layers!")
logger.info("Loading Model finished!")

'''
Load Data
'''
# print('Loading Data...')
logger.info("Loading Data...")
# configure the data loader
data_loader = data.LoadDatasetReconstruction(
        path=config.Path,
        use_label=False,
        label_filename=config.LabelFilename,
        shg_filename=config.ShgFilename,
        tbdrms_file=config.TBDrmsFilename,  # Path to the file containing TBDrms values
        tbdrms_threshold=config.TBDRMS_THRESHOLD,  # TBDrms threshold for filtering    
        transform=shg_transform,
        target_transform=label_transform
        )
################
## Split Data ##
################
# get the length of the dataset
length_dataset = len(data_loader)
logger.info(f"Size of dataset: {length_dataset}")

# get ratios of train, validation and test data
test_size = int(0.1 * length_dataset)                       # amount of test data (10%)
validation_size = int (0.1 * length_dataset)                # amount of validation data (10%) 
train_size = length_dataset - test_size - validation_size   # amount of training and validation data (80%)
logger.info(f"Size of training data:   {train_size}")
logger.info(f"Size of validation data: {validation_size}")
logger.info(f"Size of test data:       {test_size}")

# split the dataset accordingly
train_data, validation_data, test_data = random_split(data_loader, [train_size, validation_size, test_size])   # split data

# define the data loaders for training and validation
train_loader = DataLoader(train_data, batch_size = config.BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size = config.BATCH_SIZE, shuffle=False)
logger.info("Finished loading data!")

# get the number of steps per epoch
if train_size % config.BATCH_SIZE  != 0:
    NUM_STEPS_PER_EPOCH = (train_size // config.BATCH_SIZE + 1)
else:
    NUM_STEPS_PER_EPOCH = (train_size // config.BATCH_SIZE)
# get the number of total steps 
NUM_STEPS = NUM_STEPS_PER_EPOCH * config.NUM_EPOCHS

'''
Training 
'''
logger.info(f"Starting training...")
########################
## loss and optimizer ##
########################
# loss function
# define and configure the loss function
# criterion = nn.MSELoss()
criterion = loss_module.PulseRetrievalLossFunction(
        use_label = False,
        frog_error_weight= 1.0
        )

# define and configure the optimizer used
# TODO: AdamW probieren
optimizer = torch.optim.Adam(
        [   
         {'params': model.fc1.parameters()},
         {'params': model.fc2.parameters()}
        ],
        lr=1e-6,
	    weight_decay=config.WEIGHT_DECAY
	    )
#optimizer = torch.optim.SGD(
        # [   
        # { 'params': model.fc1.parameters()},
        # {'params': model.fc2.parameters()}
        # ],
        # lr=config.LEARNING_RATE,
	#momentum = 0.9)

# define and configure the scheduler for changing learning rate during training
# scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.GAMMA_SCHEDULER)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer, 
    # mode='min', 
    # factor=0.5,  # Factor by which the learning rate will be reduced (e.g., half the learning rate).
    # patience=2,  # Number of epochs with no improvement to wait before reducing the learning rate.
    # threshold=0.0001   # Threshold for measuring the new optimum.
    # )
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.5e-1,
    total_steps=NUM_STEPS
    )

# initiaize lists containing all loss values
training_losses = []
validation_losses = []
learning_rates = []

'''
Training Loop
'''
# itterate over epochs
for epoch in range(config.NUM_EPOCHS):
    # place model into training mode
    model.train()       
    # itterate over train data
    for i, (shg_matrix, header) in enumerate(train_loader):
        ###############
        ## Load Data ##
        ###############
        # send shg_matrix and label data to selected device
        shg_matrix = shg_matrix.float().to(device)
        
        ##################
        ## Forward pass ##
        ##################
        # get the predicted output from the model
        outputs = model(shg_matrix)
        # calculate the loss
        loss = criterion(
                prediction=outputs,
                label=None,
                shg_matrix=shg_matrix, 
                header=header
                )

        ###################
        ## Backward pass ##
        ###################
        # calculate gradients
        loss.backward()
	    # Gradient clipping 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad()

        # log information for current batch
        logger.info(f"Epoch {epoch+1} / {config.NUM_EPOCHS}, Step {i+1} / {NUM_STEPS_PER_EPOCH}, Loss = {loss.item():.10e}, LR = {scheduler.get_last_lr()[0]:.4e}")
        # Write loss into array
        training_losses.append(loss.item())
        
        # Step the learning rate
        scheduler.step()
        # write new learning rate in variable and save it to list
        new_lr = scheduler.get_last_lr()[0]
        learning_rates.append(new_lr)

    # After the defined unfreeze_epoch, unfreeze the earlier layers and train the whole Model
    if (epoch == config.UNFREEZE_EPOCH - 1):
        logger.info("Unfreezing earlier layers")
        
        # Unfreeze all layers
        for param in model.densenet.parameters():
            param.requires_grad = True

    '''
    Validation loop
    '''
    logger.info(f"Starting Validation for epoch {epoch+1} / {config.NUM_EPOCHS}")
    model.eval()    # put model into evaluation mode
    # 
    with torch.no_grad():   # disable gradient computation for evaluation
        # itterate over validation data
        for shg_matrix, header in validation_loader:
            ###############
            ## Load Data ##
            ###############
            # convert shg_matrix and label to float and send them to the device
            shg_matrix = shg_matrix.float().to(device)

            ##################
            ## Forward pass ## 
            ##################
            # calculate prediction
            outputs = model(shg_matrix)
            # calcultate validation loss
            validation_loss = criterion(
                    prediction=outputs, 
                    label=None,
                    shg_matrix=shg_matrix,
                    header=header
                    )
            # place validation loss into list
            validation_losses.append(validation_loss.item())

        # calculate the mean validation loss
        avg_val_loss = np.mean(validation_losses)  # calculate validation loss for this epoch
    logger.info(f"Validation Loss: {avg_val_loss:.10e}")

# save learning rate and losses to csv files
with open(config.learning_rate_filepath, 'w', newline='') as file:
    for item in learning_rates:
        file.write(f"{item}\n")
with open(config.training_loss_filepath, 'w', newline='') as file:
    for item in training_losses:
        file.write(f"{item}\n")
with open(config.validation_loss_filepath, 'w', newline='') as file:
    for item in validation_losses:
        file.write(f"{item}\n")


# plot training loss
vis.save_plot_training_loss(
        training_loss = training_losses,
        validation_loss = validation_losses,
        learning_rates = learning_rates,
        train_size = train_size // config.BATCH_SIZE,
        num_epochs = config.NUM_EPOCHS,
        filepath = f"{config.loss_plot_filepath}"
        )
logger.info("Training finished!")

# Write state_dict of model to file
torch.save(model.state_dict(), config.model_filepath)
logger.info("Saved Model")

'''
Testing Loop
'''
logger.info("Starting Test Step...")
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
test_losses = []

model.eval()
# don't calculate gradients
with torch.no_grad():
    # iterate over test data
    for shg_matrix, header in test_loader:
        # convert shg_matrix and labels to float and send them to the device
        shg_matrix = shg_matrix.float().to(device)
        
        # calculate the predicted output
        outputs = model(shg_matrix)
        # get the loss
        test_loss = criterion(
                prediction = outputs,
                label = None, 
                shg_matrix = shg_matrix,
                header = header)
        # place the loss in a list
        test_losses.append(test_loss.item())

    # calculate the mean test loss
    avg_test_loss = np.mean(test_losses)
    logger.info(f"Test Loss: {avg_test_loss:.10e}")

    # get the prediction of a random test data point and plot it
    if len(test_data) > 0:
        # get a random sample
        test_sample = random.choice(test_data)
        shg_matrix, label, header = test_sample
        # adding an extra dimension to shg_matrix and label to simulate a batch size of 1
        shg_matrix = shg_matrix.unsqueeze(0)
        # send shg_matrix to device and make prediction
        shg_matrix = shg_matrix.float().to(device)
        prediction = model(shg_matrix) 
        # send label and prediction to cpu, so that it can be plotted
        prediction = label_unscaler(prediction).cpu()
        # calculate the imaginary part of the signal and make it the shape of the label
        prediction_analytical = loss_module.hilbert(prediction.squeeze())
        prediction = torch.cat((prediction_analytical.real, prediction_analytical.imag))
        # plot
        label = torch.zeros_like(prediction)
        vis.compareTimeDomainComplex(config.random_prediction_filepath, label, prediction)

logger.info("Test Step finished!")

for handler in logger.handlers:
    handler.flush()
    handler.close()
