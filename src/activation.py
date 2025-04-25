import numpy as np
import copy


class ReLU():
    def __init__(self) -> None:
        self.cache = {
            'x': None
        }

    def __str__(self) -> str:
        out_str = 'ReLU()'
        return out_str
    
    def parameters(self):
        return None

    def forward(self, x):
        self.cache['x'] = copy.deepcopy(x)
        filt = x < 0
        x[filt] = 0
    
        return x

    def backward(self, d_last):
        filt = (self.cache['x'] < 0)
        d_last[filt] = 0
        return d_last


class Tanh():
    def __init__(self) -> None:
        self.cache = {
            'tanh_x': None
        }

    def __str__(self) -> str:
        out_str = 'Tanh()'
        return out_str
    
    def parameters(self):
        return None

    def forward(self, x):
        exp_minus_2x = np.exp(-2 * x)
        tanh_x = (1 - exp_minus_2x) / (1 + exp_minus_2x)
        self.cache['tanh_x'] = copy.deepcopy(tanh_x)
        return tanh_x

    def backward(self, d_last):
        return d_last * (1 - self.cache['tanh_x'] ** 2)