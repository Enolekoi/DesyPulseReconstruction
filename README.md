# DenseNet Femtosecond Laser Pulse Reconstruction
A DenseNet that uses FROG-traces to predict the corresponding time domain pulse

## Dependencies
- `python`
- `numpy`
- `pytorch`
- `matplotlib`
- `pandas`

## Training
To train the model supervised execute:
```$ python training.py``` 
To train the model unsupervised execute:
```$ python training_unsupervised.py``` 
The hyperparameters for training can be changed in `./config.py`.

## Files
The labview directories contains a subvi that allows the integration of the `./prediction.py` script.

Some test scripts are placed inside the `./test/` directory. They can be called by `python -m test.module_name`

Python modules are placed inside the `./modules/` directory:
- `./modules/config.py` contains the training configuration
- `./modules/constants.py` contains constants
- `./modules/data.py` contains functions for loading and transforming the SHG-matrix and labels
- `./modules/helper.py` contains various helper functions
- `./modules/loss.py` contains functions needed for the loss function
- `./modules/models.py` contains custom models used for training
- `./modules/preprocessing.py` contains functions for preprocessing experimental data
- `./modules/visualize.py` contains functions for visualizing data
 
Each training cycle creates a new directory in `./logs/` called `/log_xxx/` which contains the following files:
- The model after training get saved to the `./model.pth/` file
- The log file is saved to the `./training.log` file
- A plot of the training loss, average validation loss and learning rate is saved to the `./loss.png` file
- After training, some spectrograms from the test dataset are selected and a time domain pulses are reconstructed. They are then compared to the (label) time domain signal and saved to the, `prediction_max`, `prediction_mean`, `prediction_min` and `random_prediction.png` files
- CSV files containing learning rate, training loss and validation loss are saved to the `learning_rate.csv`, `training_loss.csv` and `validation_loss.csv` files

## Additional scripts
- `./datasetInformation.py` is a script for gathering informations regarding the dataset. It outputs the minimum and maximum values of the delay and wavelength, which will be written into a selectable location. It will also create a CSV-file which contains all TBD values and their corresponding subdirectories.
- `./find_lr.py` is a script which determines an optimal learning rate for training.
- `./prediction.py` is script for predicting a single pulse using a selected model.
- `./preprocess_script.py` is a script for preprocessing spectrograms. It is currently not used.
