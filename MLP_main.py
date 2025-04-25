import numpy as np
import copy

from mnist_data import prepare_mnist
from src.construct import get_model_from_config
from src.loss import CrossEntropy
from src.optim import SGD, Adam


# datasets
train_x, train_y = prepare_mnist('train')
b_s = 32
sample_idx = np.random.randint(0, len(train_x), (b_s, ))
sample_x, sample_y = train_x[sample_idx], train_y[sample_idx]


config = [('linear', [784, 256, False]), 
        #   ('bn1d', [256]), 
          ('relu', []), 
          ('linear', [256, 10, False])]

layers = get_model_from_config(config)
loss_fn = CrossEntropy()
parameters = [l.parameters() for l in layers]
for l in layers:
    print(l)

adam_config = dict(
    l_r = 3e-4,
    beta1 = 0.9,
    beta2 = 0.999
)

sgd_config = dict(
    l_r = 1e-3,
    momentum = 0.9
)


# optimizer = SGD(parameters, **sgd_config)
optimizer = Adam(parameters, **adam_config)


for iter_ in range(200):
    optimizer.zero_grad()
    x = copy.deepcopy(sample_x)
    y = copy.deepcopy(sample_y).flatten()

    # forward
    for l in layers:
        x = l.forward(x)
    
    loss = loss_fn.forward(x, y)
    pred_out = x.argmax(axis=-1)
    acc = np.sum(pred_out == y) / len(y)
    print(f'iter: {iter_}, loss: {loss}, acc: {acc}')

    # backward
    d_last = loss_fn.backward()
    for l in reversed(layers):
        d_last = l.backward(d_last)
    
    optimizer.step()