# ML-from-scratch

This repository implements a simple machine learning framework from scratch using NumPy. It includes modules for building neural networks, defining layers, activation functions, loss functions, and optimizers. The framework supports both fully connected (MLP) and convolutional (CNN) architectures.

## Features

- **Custom Layers**: Linear, Convolutional, Batch Normalization, Padding, Pooling, and Flatten layers.
- **Activation Functions**: ReLU and Tanh.
- **Loss Functions**: Cross-Entropy Loss.
- **Optimizers**: SGD and Adam.
- **MNIST Dataset**: Preprocessing and training examples for MLP and CNN models.

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- torchvision (for downloading the MNIST dataset)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ML-from-scratch.git
   cd ML-from-scratch
2. Install dependencies:
   ```bash
   pip install numpy torchvision
   ```

Train an MLP on MNIST
Run the MLP_main.py script:
```bash
python MLP_main.py
```

Train a CNN on MNIST
Run the conv_main.py script:
```bash
python conv_main.py
```

## Code Overview
### Layers
Defined in `src/layers.py`, this module includes:

* Linear: Fully connected layer.
* Conv2d: 2D convolutional layer.
* BatchNorm1d and BatchNorm2d: Batch normalization layers.  
* Pad4dTesnor: Padding layer.
* AvgPool: Average pooling layer.
* Flatten: Flatten layer.

### Activation Functions
Defined in `src/activation.py`, this module includes:

* ReLU: Rectified Linear Unit.
* Tanh: Hyperbolic tangent.
### Loss Functions
Defined in `src/loss.py`, this module includes:

* CrossEntropy: Cross-entropy loss for classification tasks.
### Optimizers
Defined in `src/optim.py`, this module includes:

* SGD: Stochastic Gradient Descent.
* Adam: Adaptive Moment Estimation.
### Model Construction
The `src/construct.py` module provides a utility function get_model_from_config to build models from a configuration list.

### Dataset
The `mnist_data.py` script handles downloading and preprocessing the MNIST dataset.