"""
Models Module

Module containing models
"""
import torch
import torch.nn as nn
import torchvision.models as models

'''
CustomDenseNetReconstruction()
Description:
    Custom DenseNet class for reconstructiong the time domain pulse from SHG-matrix
'''
class CustomDenseNetReconstruction(nn.Module):
    def __init__(self, num_outputs=512):
        '''
        Inputs:
            num_outputs -> [int] Number of outputs of the DenseNet
        '''
        super(CustomDenseNetReconstruction, self).__init__()
        # Load pretrained DenseNet
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        # self.densenet = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        # Get the number of features before the last layer
        num_features = self.densenet.classifier.in_features
        # Create a Layer with the number of features before the last layer and 256 outputs (2 arrays of 128 Elements)
        print(num_features)
        self.densenet.classifier = nn.Linear(num_features, 2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_outputs)
        # self.densenet.classifier = nn.Linear(num_features, num_outputs)
        self.num_outputs = num_outputs

        # initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu') # useful before relu
        nn.init.zeros_(self.fc1.bias)  # Initialize biases to zero
        nn.init.xavier_normal_(self.fc2.weight) # useful before tanh
        nn.init.zeros_(self.fc2.bias)  # Initialize biases to zero

    def forward(self, shg_matrix):
        '''
        Forward pass through the DenseNet
        Input:
            shg_matrix      -> [tensor] SHG-matrix
        Output:
            x               -> [tensor] predicted output
        '''
        # get the output of the densenet
        x = self.densenet(shg_matrix)
        x = torch.relu(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x = torch.relu(x)
        
        # use tanh activation function to scale the output to [-1, 1] and then scale it (intensity)
        x = torch.tanh(x) 
        return x
