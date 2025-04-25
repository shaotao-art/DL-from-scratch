from .layers import *
from .activation import *

def get_model_from_config(config):
    layers = []
    all_layer_type = ['linear', 
                    'pad',
                    'conv2d',
                    'relu',
                    'avgpool',
                    'flatten',
                    'bn1d',
                    'bn2d']
    for c in config:
        l_type, l_config = c
        assert l_type in all_layer_type
        if l_type == 'linear':
            layers.append(Linear(*l_config))
        if l_type == 'pad':
            layers.append(Pad4dTesnor(*l_config))
        if l_type == 'conv2d':
            layers.append(Conv2d(*l_config))
        if l_type == 'avgpool':
            layers.append(AvgPool(*l_config))
        if l_type == 'bn1d':
            layers.append(BatchNorm1d(*l_config))
        if l_type == 'bn2d':
            layers.append(BatchNorm2d(*l_config))
        if l_type == 'flatten':
            layers.append(Flatten(*l_config))
        if l_type == 'relu':
            layers.append(ReLU(*l_config))
    return layers
