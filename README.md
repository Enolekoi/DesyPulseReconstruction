# DenseNet Femtosecond Laser Pulse Reconstruction
A DenseNet that uses FROG-traces (spectrograms) to predict the corresponding time domain pulse

## Dependencies
-`python`
-`numpy`
-`pytorch`
-`matplotlib`
-`scikit-learn`
-`pandas`

## Training
To train the model execute:
```$ python main.py```

## Files
- (TODO) Models get saved to the `./models` directory
- Log files for each training step are saved to the `./logs` directory
- A plot of the training loss is saved tot the `./logs` directory
- After training a random spectrogram from the test dataset is selected and a time domain pulse is reconstructed. It is then compared to the (label) time domain signal and saved to `./random_test_prediction.png`
