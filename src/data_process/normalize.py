import torch

def Normalize(y):
    y = (y - torch.min(y)) / (torch.max(y) - torch.min(y))
    return y

def np_Normalize(y):
    y = (y - y.min()) / (y.max() - y.min())
    return y