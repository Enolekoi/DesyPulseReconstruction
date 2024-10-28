# DenseNet Femtosecond Laser Pulse Reconstruction
A DenseNet that uses FROG-traces (spectrograms) to predict the corresponding time domain pulse

## Dependencies
- `python`
- `numpy`
- `pytorch`
- `matplotlib`
- `pandas`

## Training
To train the model execute:
```$ python training.py```
Most parameters for training can be changed in `./config.py`.

## Files
The labview and additional directories contain not up to date files, that will be replaced later.

Some test scripts are placed inside the `./test/` directory. They can be called by `python -m test.module_name`

Python modules are placed inside the `./modules/` directory:
- `./modules/config.py` contains the training configuration
- `./modules/constants.py` contains constants
- `./modules/data.py` contains functions for loading and transforming the SHG-matrix and labels
- `./modules/helper.py` contains various helper functions
- `./modules/loss.py` contains functions needed for the loss function
- `./modules/models.py` contains custom models used for training
- `./modules/preprocessing.py` contains functions for preprocessing experimental data
- `./modules/tbdrms.py` contains functions and classes for predicting the TBDrms values
- `./modules/visualize.py` contains functions for visualizing data
 

Each Training creates a new directory in `./logs/` called `/training_xxx/` which contains the following files:
- The model after training get saved to the `./model.pth/` file
- The log file is saved to the `./training.log` file
- A plot of the training loss, average validation loss and learning rate is saved to the `./loss.png` file
- After training, a random spectrogram from the test dataset is selected and a time domain pulse is reconstructed. It is then compared to the (label) time domain signal and saved to the `random_prediction.png` file
- CSV files containing learning rate, training loss and validation loss are saved to the `learning_rate.csv`, `training_loss.csv` and `validation_loss.csv` files

The training dataset is expected to be in the `/mnt/data/desy/frog_simulated/grid_256_v3/` directory (this can be changed in `./config.py`)
