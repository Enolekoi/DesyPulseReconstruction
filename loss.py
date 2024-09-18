'''
loss.py Module

Module containing functions used for loss functions
'''

import torch
import torch.nn as nn 

'''
PulseRetrievalLossFunction():
Loss function for pulse retrieval
'''
class PulseRetrievalLossFunction(nn.Module):
    def __init__(self, penalty_factor=2.0, threshold=0.01):
        '''
        Initialization
        Inputs:
            weight_factor   -> Factor by which the loss is multiplied, when the label is greater than the threshold [float]
            threshold       -> Label value over which the higher weights get multiplied with the loss [float]
        '''
        super(PulseRetrievalLossFunction, self).__init__()

        self.penalty_factor = penalty_factor
        self.threshold = threshold

    def forward(self, prediction, label):
        # print(f"prediction size = {predictions}")
        # print(f"label size = {labels}")
        
        # get the number of batches, as well as the shape of the labels
        batch_size, num_elements = label.shape
        # get half of elements
        half_size = num_elements // 2

        # get real and imaginary parts of labels and predictions
        label_real = label[:, :half_size]
        label_imag = label[:, half_size:]

        prediction_real = prediction[:, :half_size]
        prediction_imag = prediction[:, half_size:]

        label_phase =      torch.atan2(label_imag, label_real)
        prediction_phase = torch.atan2(prediction_imag, prediction_real)

        label_intensity = label_real**2 + label_imag**2
        prediction_intensity = prediction_real**2 + prediction_imag**2
        
        # initialize loss
        loss = 0.0
        # Loop over each batch
        for i in range(batch_size):
            # Create masks for all absolute values higher than the threshold
            mask_real_threshold = abs(label_real[i]) > self.threshold
            mask_imag_threshold = abs(label_imag[i]) > self.threshold
            
            # if any real value is greater than the threshold
            if torch.any(mask_real_threshold):
                # get the first and last index, where a value is greater than the threshold
                first_significant_idx_real = torch.nonzero(mask_real_threshold).min().item()
                last_significant_idx_real = torch.nonzero(mask_real_threshold).max().item()
            else:
                first_significant_idx_real = 0
                last_significant_idx_real = half_size - 1

            # if any imaginary value is greater than the threshold
            if torch.any(mask_imag_threshold):
                # get the first and last index, where a value is greater than the threshold
                first_significant_idx_imag = torch.nonzero(mask_imag_threshold).min().item()
                last_significant_idx_imag = torch.nonzero(mask_imag_threshold).max().item()
            else:
                first_significant_idx_imag = 0
                last_significant_idx_imag = half_size - 1

            # Calculate MSE for the real and imaginary part
            mse_real = (prediction_real[i] - label_real[i]) ** 2
            mse_imag = (prediction_imag[i] - label_imag[i]) ** 2

            # Apply penalty for values before the first significant index and after the last
            mse_real[:first_significant_idx_real] *= self.penalty_factor
            mse_real[last_significant_idx_real + 1:] *= self.penalty_factor
            mse_imag[:first_significant_idx_imag] *= self.penalty_factor
            mse_imag[last_significant_idx_imag + 1:] *= self.penalty_factor
            
            mse_intensity = (prediction_intensity[i] - label_intensity[i]) ** 2
            mse_phase = (prediction_phase[i] - label_phase[i]) ** 2

            # Add to total loss
            loss += mse_real.mean() + mse_imag.mean() + 50*mse_intensity.mean() + 0*mse_phase.mean()
        # devide by batch size 
        loss = loss / batch_size

        return loss

def createSHGmat(yta, Ts, wCenter):
    N = len(yta)
    # create a tensor storing indicies starting with -N/2 to N/2
    start = -N // 2
    end = N // 2    
    delayIdxVec = torch.arange(start, end, dtype=torch.float32)

    # calculate shift factor
    shiftFactor = torch.exp(-1j * 2 * wCenter * Ts * delayIdxVec)

    shgMat = torch.zeros((N, N), dtype=torch.complex64)

    def fftshift(x):
        return torch.fft.fftshift(x)
    
    def circshift(x, shift):
        shift = shift % x.size[0]
        return torch.roll(x, shifts=shift, dims=0)

    for (matIdx, delayIdx) in enumerate(delayIdxVec):
        ytaShifted = circshift(yta, delayIdx)
        fft_yta = torch.fft.fft(fftshift(yta* ytaShifted * shiftFactor))
        shgMat[matIdx, :] = Ts * fftshift(fft_yta)
