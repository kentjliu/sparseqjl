import torch
import torch.nn as nn

DEV = torch.device('cuda')

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    """ 
    Recursively searches through a Pytorch nn.Module to find all the layers of a specific type.

    Returns a dictionary mapping the layer names to the layer objects themselves.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
