import numpy as np
# reference page:
# sgd: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
# adam: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

class SGD:
    def __init__(self, parameters, l_r, momentum) -> None:
        assert isinstance(parameters, list)
        self.l_r = l_r
        self.parameters = parameters
        self.momentum = momentum
        self.m = self.init_m(self.parameters)
        
    def init_m(self, model_param_lst):
        m = []
        for p in model_param_lst:
            if p is not None:
                tmp_dict = {}
                assert isinstance(p, dict)
                for k, v in p.items():
                    tmp_dict[k] = np.zeros_like(v.param)
                m.append(tmp_dict)
            else:
                m.append(None)
        return m
    
    def zero_grad(self):
        for p in self.parameters:
            if p is not None:
                assert isinstance(p, dict)
                for k, v in p.items():
                    p[k].gradient *= 0
                    
    def update_m(self, model_param_lst):
        # NOTE: in the first step, m <- grad
        for layer_idx, p in enumerate(model_param_lst):
            if p is not None:
                assert isinstance(p, dict)
                for k, v in p.items():
                    self.m[layer_idx][k] = self.momentum * self.m[layer_idx][k] + v.gradient
                    
    
    def step(self):
        self.update_m(self.parameters)
        for layer_idx, p in enumerate(self.parameters):
            if p is not None:
                assert isinstance(p, dict)
                for k, v in p.items():
                    self.parameters[layer_idx][k].param -= self.l_r * self.m[layer_idx][k]
        
        
class Adam:
    def __init__(self, parameters, l_r, beta1, beta2, eps=1e-8) -> None:
        assert isinstance(parameters, list)
        self.l_r = l_r
        self.parameters = parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.m = self.init_m_v(self.parameters)
        self.v = self.init_m_v(self.parameters)
        self.t = 0
        

    def init_m_v(self, model_param_lst):
        m = []
        for p in model_param_lst:
            if p is not None:
                tmp_dict = {}
                assert isinstance(p, dict)
                for k, v in p.items():
                    tmp_dict[k] = np.zeros_like(v.param)
                m.append(tmp_dict)
            else:
                m.append(None)
        return m

        
    def update_m_v(self, model_p_lst):
        self.t += 1
        for layer_idx, p in enumerate(model_p_lst):
            if p is not None:
                assert isinstance(p, dict)
                for k, v in p.items():
                    # NOTE: should store original m, v, which is not corrected
                    self.m[layer_idx][k] = self.beta1 * self.m[layer_idx][k] + (1 - self.beta1) * v.gradient
                    self.v[layer_idx][k] = self.beta2 * self.v[layer_idx][k] + (1 - self.beta2) * v.gradient ** 2
    
    def zero_grad(self):
        for layer_idx, p in enumerate(self.parameters):
            if p is not None:
                assert isinstance(p, dict)
                for k, v in p.items():
                    self.parameters[layer_idx][k].gradient *= 0
    
    def step(self):
        self.update_m_v(self.parameters)
        for layer_idx, p in enumerate(self.parameters):
            if p is not None:
                assert isinstance(p, dict)
                for k, v in p.items():
                    m = self.m[layer_idx][k] / (1 - self.beta1 ** self.t)
                    v = self.v[layer_idx][k] / (1 - self.beta2 ** self.t)
                    self.parameters[layer_idx][k].param -= self.l_r * m / (np.sqrt(v) + self.eps)
                    