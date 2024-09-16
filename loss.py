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

    def forward(self, predictions, labels):
        # print(f"prediction size = {predictions}")
        # print(f"label size = {labels}")
        
        # get the number of batches, as well as the shape of the labels
        batch_size, num_elements = labels.shape
        # get half of elements
        half_size = num_elements // 2

        # get real and imaginary parts of labels and predictions
        labels_real = labels[:, :half_size]
        labels_imag = labels[:, half_size:]

        predictions_real = predictions[:, :half_size]
        predictions_imag = predictions[:, half_size:]
        
        # initialize loss
        loss = 0.0
        # Loop over each batch
        for i in range(batch_size):
            # Create masks for all absolute values higher than the threshold
            mask_real_threshold = abs(labels_real[i]) > self.threshold
            mask_imag_threshold = abs(labels_imag[i]) > self.threshold
            
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
            mse_real = (predictions_real[i] - labels_real[i]) ** 2
            mse_imag = (predictions_imag[i] - labels_imag[i]) ** 2

            # Apply penalty for values before the first significant index and after the last
            mse_real[:first_significant_idx_real] *= self.penalty_factor
            mse_real[last_significant_idx_real + 1:] *= self.penalty_factor
            mse_imag[:first_significant_idx_imag] *= self.penalty_factor
            mse_imag[last_significant_idx_imag + 1:] *= self.penalty_factor
            
            # Add to total loss
            loss += mse_real.mean() + mse_imag.mean()
        # devide by batch size 
        loss = loss / batch_size

        return loss


def createSHGmat(analytical_time_signal, sampling_time, w_center):
    N = len(analytical_time_signal)
    delay_index_vector = (-N // 2) 
    
