import numpy as np
import copy

__all__ = [
    'Linear',
    'BatchNorm1d',
    'BatchNorm2d',
    'Pad4dTesnor',
    'AvgPool',
    'Flatten',
    'Conv2d',
]

class data_w:
    def __init__(self) -> None:
        self.param = None
        self.gradient = None
        
        
class Linear():
    """Expected input shape: (b_s, in_dim)
    weight: (in_dim, out_dim)
    bias: (out_dim, )
    output: (b_s, out_dim)"""
    def __init__(self, in_dim, out_dim, use_bias=True) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        
        if use_bias:
            self.params = {
                'weight': data_w(),
                'bias': data_w()
            }
        else:
            self.params = {
                'weight': data_w(),
            }

        self.cache = {'x': None}
        
        self._init_params()

    def _init_params(self):
        # kaiming-norm
        # reference: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        gain = np.sqrt(2) # assume relu
        std = gain / np.sqrt(self.in_dim) # forward pass mode
        self.params['weight'].param = np.random.randn(self.in_dim, self.out_dim) * std
        self.params['weight'].gradient = np.zeros_like(self.params['weight'].param)
        
        if self.use_bias:
            self.params['bias'].param= np.zeros((self.out_dim, ))
            self.params['bias'].gradient = np.zeros_like(self.params['bias'].param)

    def __str__(self) -> str:
        out_str = f"Linear(in_dim={self.in_dim}, out_dim={self.out_dim}) #params: {self.params['weight'].param.size + self.params['bias'].param.size if self.use_bias else self.params['weight'].param.size}"
        return out_str

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, f'Linear input dim should equal to in dim, get {x.shape[-1]}, expect {self.in_dim}'
        self.cache['x'] = copy.deepcopy(x)
        out = x @ self.params['weight'].param
        if self.use_bias:
            out += self.params['bias'].param
        return out

    def backward(self, d_last):
        # compute gradient to self params
        # also need to compute gradient for next node
        assert d_last.shape[-1] == self.out_dim, f'Linear d_last dim should equal to out dim, get {d_last.shape[-1]}, expect {self.out_dim}'
        self.params['weight'].gradient += self.cache['x'].T @ d_last
        if self.use_bias:
            self.params['bias'].gradient += np.sum(d_last, axis=0)
        dx = d_last @ self.params['weight'].param.T
        return dx

    def parameters(self) -> dict:
        return self.params
    
    
class BatchNorm1d():
    def __init__(self, feat_dim, eps=1e-5, track_running_state=True, momentum=0.1) -> None:
        self.feat_dim = feat_dim
        self.eps = eps
        self.track_running_state = track_running_state
        self.momentum = momentum

        self.running_mean = None
        self.running_std = None

        self.params = {
            'gamma': data_w(),
            'beta': data_w(),
        }

        self.cache = {
            'sub_mean': None,
            'inverse_std': None,
            'div_std': None
        }

        self._init_params()

    def _init_params(self):
        self.params['gamma'].param = np.ones((1, self.feat_dim))
        self.params['beta'].param = np.zeros((1, self.feat_dim))

        self.params['gamma'].gradient = np.zeros((1, self.feat_dim))
        self.params['beta'].gradient = np.zeros((1, self.feat_dim))

    def __str__(self) -> str:
        out_str = f"Batchnorm1d({self.feat_dim}) #params: {self.feat_dim * 2}"
        return out_str

    def forward(self, x, train=True):
        """x: (b_s, feat_dim)
        running mean and std: (1, feat_dim)
        gamma, beta: (1, feat_dim)"""
        if train == True:
            mean = x.mean(axis=0, keepdims=True)
            sub_mean = x - mean
            inverse_std = 1 / np.sqrt((sub_mean ** 2).mean(axis=0, keepdims=True) + self.eps)
            
            if self.track_running_state:
                if self.running_mean is not None:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_std = (1 - self.momentum) * self.running_std + self.momentum * 1 / inverse_std
                else:
                    self.running_mean = mean
                    self.running_std = 1 / inverse_std
            
            div_std = sub_mean * inverse_std
            out = div_std * self.params['gamma'].param + self.params['beta'].param
            self.cache['sub_mean'] = copy.deepcopy(sub_mean)
            self.cache['inverse_std'] = copy.deepcopy(inverse_std)
            self.cache['div_std'] = copy.deepcopy(div_std)
            return out
        else:
            div_std = (x - self.running_mean) / self.running_mean
            out = div_std * self.params['gamma'].param + self.params['beta'].param
            return out
            

    def backward(self, d_last):
        b_s, _ = d_last.shape
        self.params['beta'].gradient += np.sum(d_last, axis=0, keepdims=True)
        self.params['gamma'].gradient += (d_last * self.cache['div_std']).sum(axis=0, keepdims=True)
        d_div_std = d_last * self.params['gamma'].param
        d_sub_mean_1 = d_div_std * self.cache['inverse_std']
        tmp = -0.5 * (self.cache['inverse_std'] ** 3) * (d_div_std * self.cache['sub_mean']).sum(axis=0, keepdims=True)
        d_sub_mean_2 = np.repeat(tmp * (1 / b_s), repeats=b_s, axis=0) * 2 * self.cache['sub_mean']
        d_sub_mean = d_sub_mean_1 + d_sub_mean_2

        d_x_2 = np.sum(d_sub_mean * -1, axis=0, keepdims=True) * np.ones_like(d_last) / b_s
        d_x = d_sub_mean + d_x_2
        return d_x
    
    def parameters(self):
        return self.params
    
class BatchNorm2d():
    """use bn1d to compute bn2d
    bn2d input: (b_s, c, h, w)
    bn1d input: (b_s, c)
    so reshape to (b_s * h * w, c) to compute bn1d"""
    def __init__(self, feat_dim, eps=1e-5, track_running_state=True, momentum=0.1) -> None:
        self.feat_dim = feat_dim
        self.eps = eps
        self.track_running_state = track_running_state
        self.momentum = momentum
        
        self.inner_bn = BatchNorm1d(feat_dim=feat_dim,
                                    eps=eps,
                                    track_running_state=track_running_state,
                                    momentum=momentum)

    def __str__(self) -> str:
        out_str = f"Batchnorm2d({self.feat_dim}) #params: {self.feat_dim * 2}"
        return out_str

    def forward(self, x, train=True):
        assert len(x.shape) == 4
        n, c, h, w = x.shape
        x = np.transpose(x, (0, 2, 3, 1)).reshape(-1, c)
        out = self.inner_bn.forward(x, train)
        out = np.transpose(out.reshape(n, h, w, c), (0, 3, 1, 2))
        return out
            

    def backward(self, d_last):
        assert len(d_last.shape) == 4
        n, c, h, w = d_last.shape
        d_last = np.transpose(d_last, (0, 2, 3, 1)).reshape(-1, c)
        d_last = self.inner_bn.backward(d_last)
        dout = np.transpose(d_last.reshape(n, h, w, c), (0, 3, 1, 2))
        return dout
    
    def parameters(self):
        return self.inner_bn.parameters()



class Pad4dTesnor():
    """x shape: (b_s, c, h, w)
    p: padding size
    output shape: (b_s, c, h + 2 * p, w + 2 * p)"""
    def __init__(self, p) -> None:
        self.p = p

    def __str__(self) -> str:
        out_str = f"Pad(p={self.p})"
        return out_str

    def forward(self, x):
        x = np.pad(x, ((0, 0), (0, 0), (self.p, self.p), (self.p, self.p)))
        return x

    def backward(self, d_last):
        return d_last[..., self.p: -self.p, self.p: -self.p]

    def parameters(self):
        return None
    
    
class AvgPool():
    """x shape: (b_s, c, h, w)
    k: kernel size
    output shape: (b_s, c, h // k, w // k)"""
    def __init__(self, k) -> None:
        self.k = k

    def __str__(self) -> str:
        out_str = f"AvgPool(k={self.k})"
        return out_str

    def forward(self, x):
        assert x.shape[-1] % self.k == 0, f'AvgPool input dim should be divisible by k, using padding to fix this'
        assert x.shape[-2] % self.k == 0, f'AvgPool input dim should be divisible by k, using padding to fix this'
        assert len(x.shape) == 4
        b, c, in_h, in_w = x.shape
        out_h, out_w = in_h // self.k, in_w // self.k
        out = np.zeros((b, c, out_h, out_w))
        for h in range(out_h):
            for w in range(out_w):
                x_cut = x[..., h*self.k: (h+1)*self.k, w*self.k:(w+1)*self.k]
                out[..., h, w] = np.mean(x_cut, axis=(2, 3))
        return out
        

    def backward(self, d_last):
        b, c, h, w = d_last.shape
        out = np.zeros((b, c, h*self.k, w*self.k))
        for i in range(self.k):
            for j in range(self.k):
                out[..., i::self.k, j::self.k] = d_last
        return out


    def parameters(self):
        return None
    
    
class Flatten():
    """x shape: (b_s, c, 1, 1)
    output shape: (b_s, c)"""
    def __init__(self) -> None:
        pass
        
    def __str__(self) -> str:
        out_str = f"Flatten()"
        return out_str

    def forward(self, x):
        assert len(x.shape) == 4
        return x.squeeze()
        
    def backward(self, d_last):
        assert len(d_last.shape) == 2
        return np.expand_dims(d_last, axis=(2, 3))

    def parameters(self):
        return None 
    
    
class Conv2d():
    """x shape: (b_s, in_channel, h, w)
    kernels shape: (out_channel, in_channel, k, k)
    bias shape: (out_channel, 1, 1)
    output shape: (b_s, out_channel, h', w')"""
    def __init__(self, in_channel, out_channel, k, s, use_bias=False) -> None:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k = k
        self.s = s

        if use_bias:
            self.params = {
                'weight': data_w(),
                'bias': data_w()
            }
        else:
            self.params = {
                'weight': data_w(),
            }

        self.cache = {'x': None}

        self._init_params()

    def _init_params(self):
        # kaiming norm
        # reference: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        gain = np.sqrt(2) # assume relu
        std = gain / np.sqrt(self.in_channel * self.k * self.k) # forward pass mode
        self.params['weight'].param = np.random.randn(self.out_channel, self.in_channel, self.k, self.k) * std
        self.params['weight'].gradient = np.zeros_like(self.params['weight'].param)
        
        if 'bias' in self.params:
            self.params['bias'].param= np.zeros((self.out_channel, 1, 1))
            self.params['bias'].gradient = np.zeros_like(self.params['bias'].param)

    def __str__(self) -> str:
        out_str = f"Conv2d(in_channel={self.in_channel}, out_channel={self.out_channel}, k={self.k}, s={self.s} #params: {self.params['weight'].param.size + self.params['bias'].param.size if 'bias' in self.params else self.params['weight'].param.size}"
        return out_str
    
    def conv_forward(self, paded_inp, kernels, s):
        b_s, in_channel, paded_inp_h, paded_inp_w = paded_inp.shape
        out_channel, in_channel_, kernel_h, kernel_w = kernels.shape
        assert in_channel == in_channel_
        out_h = (paded_inp_h - kernel_h) // s + 1
        out_w = (paded_inp_w - kernel_w) // s + 1
        assert (paded_inp_h - kernel_h) / s + 1 == out_h
        assert (paded_inp_w - kernel_w) / s + 1== out_w
        out = np.zeros((b_s, out_channel, out_h, out_w))
        for c in range(out_channel):
            for h in range(out_h):
                for w in range(out_w):
                    out[:, c, h, w] = np.sum(paded_inp[..., h*s: h*s+self.k, w*s: w*s+self.k] * kernels[c], axis=(1, 2, 3))
        if 'bias' in self.params:
            out += self.params['bias'].param
        return out
    
    def conv_backward(self, d_last, inp, kernels, s):
        d_x = np.zeros_like(inp)
        d_kernel = np.zeros_like(kernels)
        b, out_channel, out_h, out_w = d_last.shape
        for c in range(out_channel):
            for h in range(out_h):
                for w in range(out_w):
                    inp_cut = inp[..., h*s: h*s+self.k, w*s: w*s+self.k]
                    d_last_cut = np.expand_dims(d_last[:, c, h, w], axis=(1, 2, 3))
                    d_kernel[c] += np.sum(inp_cut * d_last_cut, axis=0)
                    
                    d_x_cut = d_x[..., h*s: h*s+self.k, w*s: w*s+self.k]
                    kernel_cut = kernels[c]
                    d_x_cut += d_last_cut * kernel_cut
        d_bias = None
        if 'bias' in self.params:
            d_bias = np.sum(d_last, axis=(0, 2, 3), keepdims=True).reshape(self.out_channel, 1, 1)
        return d_x, d_kernel, d_bias

    def forward(self, x):
        self.cache['x'] = copy.deepcopy(x)
        return self.conv_forward(paded_inp=x,
                                 kernels=self.params['weight'].param,
                                 s=self.s)
        

    def backward(self, d_last):
        d_x, d_kernel, d_bias = self.conv_backward(d_last=d_last,
                                  inp=self.cache['x'],
                                  kernels=self.params['weight'].param,
                                  s=self.s)
        self.params['weight'].gradient += d_kernel
        if 'bias' in self.params:
            self.params['bias'].gradient += d_bias
        return d_x
        
    def parameters(self):
        return self.params