'''
test.py Script

Script used for testing
'''
import labview
import torch

N = 5
spec = torch.rand(N,N)

labview.plotSpectrogram(spec)
