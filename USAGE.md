# Usage Guide

This guide provides examples and explanations for using the Multidimensional Neural Networks (MNN) library.

## Basic Concepts

The MNN architecture processes tensors along multiple dimensions, allowing for more efficient computation compared to traditional approaches. Key concepts include:

- **Shape**: The dimensions of your input tensor (excluding batch dimension)
- **View**: How parameters are shared across dimensions ('shared' or 'separate')
- **Execution**: How dimensions are processed ('parallel' or 'sequential')
- **Sequential Order**: The order in which dimensions are processed when using sequential execution ('ascending' or 'descending')

## TensorFlow Implementation

### Basic Usage

```python
import tensorflow as tf
import numpy as np
from layer import MNN

# Define the shape of your input tensor (excluding batch dimension)
shape = [16, 16, 16, 16]

# Create sample data
batch_size = 32
x = np.random.random([batch_size] + shape)
y = np.random.random([batch_size] + shape)

# Create a model with a multidimensional layer
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape),
    MNN(shape=shape, view='separate', execution='parallel')
])

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=5, batch_size=8)
```

### Parameter Sharing Options

```python
# Shared parameters across all dimensions
mnn_shared = MNN(shape=shape, view='shared')

# Separate parameters for each dimension
mnn_separate = MNN(shape=shape, view='separate')

# Mixed sharing (specify for each dimension)
views = ['shared', 'separate', 'shared', 'separate']
mnn_mixed = MNN(shape=shape, view=views)
```

### Execution Modes

```python
# Parallel execution (process all dimensions simultaneously)
mnn_parallel = MNN(shape=shape, view='shared', execution='parallel')

# Sequential execution (process dimensions one after another)
mnn_sequential = MNN(shape=shape, view='shared', execution='sequential')

# Sequential with custom order
mnn_desc = MNN(shape=shape, view='shared', execution='sequential', sequential_order='descending')
```

## PyTorch Implementation

### Basic Usage

```python
import torch
import numpy as np
from layer import MNN

# Define the shape of your input tensor (excluding batch dimension)
shape = [16, 16, 16, 16]

# Create sample data
batch_size = 32
x = torch.randn(batch_size, *shape)
y = torch.randn(batch_size, *shape)

# Create a multidimensional layer
mnn_layer = MNN(shape=shape, view='separate', execution='parallel')

# Define a simple model
class MNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mnn = mnn_layer
        
    def forward(self, x):
        return self.mnn(x)

# Create model, loss function, and optimizer
model = MNNModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## Resizing Layer

The library also provides a `resizing_layer` class that allows you to change the size of specific dimensions:

```python
from layer import resizing_layer

# Resize a specific dimension
shape = [16, 16, 16]
axis = 1  # The dimension to resize
output_shape = 32  # New size for that dimension
sharing = True  # Whether to share parameters

resize = resizing_layer(shape=shape, axis=axis, output_shape=output_shape, sharing=sharing)
```

## Advanced Usage

### Combining Multiple MNN Layers

```python
# TensorFlow example
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape),
    MNN(shape=shape, view='shared', execution='parallel'),
    MNN(shape=shape, view='separate', execution='sequential')
])

# PyTorch example
class AdvancedMNNModel(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.mnn1 = MNN(shape=shape, view='shared', execution='parallel')
        self.mnn2 = MNN(shape=shape, view='separate', execution='sequential')
        
    def forward(self, x):
        x = self.mnn1(x)
        x = self.mnn2(x)
        return x
```

## Performance Considerations

- **Shared vs. Separate**: Shared parameters use less memory but may have less expressive power
- **Parallel vs. Sequential**: Parallel execution is typically faster but sequential may capture different relationships
- **Dimension Size**: Larger dimensions require more computation; consider reshaping to more balanced dimensions

For more detailed information about the architecture and theory, please refer to the [paper](paper.md).
