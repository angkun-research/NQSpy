import torch

def total_squared_loss(output, target):
    return torch.sum((output - target) ** 2)

def fidelity_loss(output, target):
    return -(torch.sum(output * target)) ** 2

#def energy_loss(output, target, H):
