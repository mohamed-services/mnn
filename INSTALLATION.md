# Installation Guide

This guide provides instructions for installing and setting up the Multidimensional Neural Networks (MNN) library.

## Requirements

- Python 3.6 or higher
- TensorFlow 2.0+ or PyTorch 1.0+ (depending on which implementation you want to use)
- NumPy

## Installation

### Option 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/mohamed-services/nn.git
cd nn

# Install dependencies
pip install -r requirements.txt  # Note: requirements.txt will be added in future updates
```

### Option 2: Direct Download

You can also download the `layer.py` file directly and include it in your project.

## TensorFlow Setup

To use the TensorFlow implementation, ensure you have TensorFlow installed:

```bash
pip install tensorflow>=2.0.0
```

Example usage:

```python
import tensorflow as tf
from layer import MNN as TF_MNN

# Define the shape of your input tensor (excluding batch dimension)
shape = [16, 16, 16]

# Create a multidimensional layer with shared parameters
mnn_layer = TF_MNN(shape=shape, view='shared', execution='parallel')

# Use the layer in a model
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape),
    mnn_layer
])
```

## PyTorch Setup

To use the PyTorch implementation, ensure you have PyTorch installed:

```bash
pip install torch>=1.0.0
```

Example usage:

```python
import torch
from layer import MNN as Torch_MNN

# Define the shape of your input tensor (excluding batch dimension)
shape = [16, 16, 16]

# Create a multidimensional layer with separate parameters
mnn_layer = Torch_MNN(shape=shape, view='separate', execution='sequential')

# Use the layer in a model
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mnn = mnn_layer
        
    def forward(self, x):
        return self.mnn(x)
```

## Troubleshooting

If you encounter any issues during installation or setup:

1. Ensure you have the correct versions of Python, TensorFlow, or PyTorch installed
2. Check that NumPy is properly installed
3. If you're using GPU acceleration, make sure your CUDA and cuDNN versions are compatible with your TensorFlow or PyTorch version

For further assistance, please open an issue on the GitHub repository.
