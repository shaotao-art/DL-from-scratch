from torchvision.datasets import MNIST
import numpy as np

def prepare_mnist(split, download=True):
    dataset = MNIST('./data', train=True if split == 'train' else False, download=download)
    x = [np.array(dataset[x][0], dtype=np.float32) for x in range(len(dataset))]
    x = [img.reshape(1, -1) for img in x]
    x = np.concatenate(x, axis=0)
    x = (x - x.mean()) / (x.std() + 1e-6)

    y = [np.array(dataset[x][1]) for x in range(len(dataset))]
    y = [label.reshape(1, -1) for label in y]
    y = np.concatenate(y)
    return x, y