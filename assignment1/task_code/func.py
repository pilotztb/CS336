import torch
import torch.nn as nn

def silu(x):
    return x * torch.sigmoid(x)