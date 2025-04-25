import numpy as np
import copy

class CrossEntropy():
    def __init__(self) -> None:
        super().__init__()
        self.cache = {
            'prob': None,
            'y': None
        }

    def __str__(self) -> str:
        out_str = 'CrossEntropyLoss()'
        return out_str

    def forward(self, x, y):
        assert len(y.shape) == 1
        b_s = x.shape[0]

        shifted_x = x - np.max(x, axis=1, keepdims=True)
        prob = np.exp(shifted_x) / np.exp(shifted_x).sum(axis=1, keepdims=True)
        loss = - np.log(np.clip(prob[np.arange(b_s), y], a_min=1e-5, a_max=1 - 1e-5)).mean()
        self.cache['prob'] = copy.deepcopy(prob)
        self.cache['y'] = copy.deepcopy(y)
        return loss


    def backward(self):
        prob = self.cache['prob']
        y = self.cache['y']
        b_s, _ = prob.shape
        dL = copy.deepcopy(prob)
        dL[np.arange(len(prob)), y] = dL[np.arange(len(prob)), y] - 1
        dL /= b_s
        return dL
    
    def parameters(self):
        return None