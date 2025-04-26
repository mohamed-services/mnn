
# tensorflow implementation
import tensorflow as tf
import keras
import string

class MNN_tf(keras.layers.Layer):
    def __init__(self, shape, view, execution='parallel', sequential_order: str='ascending', **kwargs):
        super().__init__()
        self.shape = shape
        self.execution = execution.lower()
        if self.execution not in ['parallel', 'sequential']:
            raise ValueError('execution value must be parallel or sequential')
        self.sequential_order = sequential_order.lower()
        if type(view) == str:
            view = [view.lower() for i in shape]
        elif len(view) == 1:
            view = [view[0].lower() for i in shape]
        self.w = {}
        for axis in range(-len(self.shape),0):
            if view[axis] == 'shared':
                in_shape = [shape[axis]]
            elif view[axis] == 'separate':
                in_shape = list(shape)
            else:
                raise ValueError('view value missing or invalid, valid view values shared or separate')
            self.w[axis] = self.add_weight(shape=in_shape+[shape[axis]], **kwargs)
    def axis_call(self, x, w, axis):
        equation_x = string.ascii_letters[:len(x.shape)]
        if len(w.shape) == 2: #shared
            equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
        else: #separate
            equation_w = equation_x[1-len(w.shape):] + string.ascii_letters[len(equation_x)]
        equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
        equation = equation_x + ',' + equation_w + '->' + equation_o
        print(equation)
        return tf.einsum(equation, x, w)
    def call(self, x):
        if self.execution == 'parallel':
            sub_layer = []
            for axis in self.w:
                sub_layer.append(axis_call(x, self.w[axis], axis))
            x = sum(sub_layer)
        elif self.execution == 'sequential':
            order = self.w if self.sequential_order == 'ascending' else reversed(self.w)
            for axis in order:
                x = axis_call(x, self.w[axis], axis)
        return x

class resizing_layer_tf(keras.layers.Layer):
    def __init__(self, shape, axis: int, output_shape: int, sharing: bool, **kwargs):
        super().__init__()
        self.shape = shape
        self.axis = axis
        self.w = {}
        if sharing == True:
            self.w[axis] = self.add_weight(shape=[shape[axis], output_shape])
        elif sharing == False:
            self.w[axis] = self.add_weight(shape=list(shape) + [output_shape])
    def axis_call(self, x, w, axis):
        equation_x = string.ascii_letters[:len(x.shape)]
        if len(w.shape) == 2: #shared
            equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
        else: #separate
            equation_w = equation_x[1-len(w.shape):] + string.ascii_letters[len(equation_x)]
        equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
        equation = equation_x + ',' + equation_w + '->' + equation_o
        print(equation)
        return tf.einsum(equation, x, w)
    def call(self, x):
        x = axis_call(x, self.w[self.axis], self.axis)
        return x


if __name__ == '__main__':
    import numpy as np
    import time
    shape = [16,16,16,16]
    x = np.random.random([64]+shape)
    y = np.random.random([64]+shape)
    model = keras.Sequential([
        keras.layers.InputLayer(shape),
        MNN_tf(shape, 'separate')
    ])
    model.compile('adam', 'mse')
    model.summary()
    t = time.time()
    model.fit(x, y, epochs=16)
    t = time.time() - t
    print(t)

# add preinitialized weights in the init


# torch implementation
import torch
import string

class MNN_torch(torch.nn.Module):
    def __init__(self, shape, view, execution='parallel', sequential_order: str='ascending', **kwargs):
        super().__init__()
        self.shape = shape
        self.execution = execution.lower()
        if self.execution not in ['parallel', 'sequential']:
            raise ValueError('execution value must be parallel or sequential')
        self.sequential_order = sequential_order.lower()
        if type(view) == str:
            view = [view.lower() for i in shape]
        elif len(view) == 1:
            view = [view[0].lower() for i in shape]
        self.w = {}
        for axis in range(-len(self.shape),0):
            if view[axis] == 'shared':
                in_shape = [shape[axis]]
            elif view[axis] == 'separate':
                in_shape = list(shape)
            else:
                raise ValueError('view value missing or invalid, valid view values shared or separate')
            param = torch.nn.Parameter(torch.empty(in_shape+[shape[axis]]))
            torch.nn.init.xavier_uniform_(param)
            param_name = f'weight_{axis}'
            self.register_parameter(param_name, param)
            self.w[axis] = param
    def axis_call(self, x, w, axis):
        equation_x = string.ascii_letters[:len(x.shape)]
        if len(w.shape) == 2: #shared
            equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
        else: #separate
            equation_w = equation_x[1-len(w.shape):] + string.ascii_letters[len(equation_x)]
        equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
        equation = equation_x + ',' + equation_w + '->' + equation_o
        #print(equation)
        return torch.einsum(equation, x, w)
    def forward(self, x):
        if self.execution == 'parallel':
            sub_layer = []
            for axis in self.w:
                sub_layer.append(axis_call(x, self.w[axis], axis))
            x = sum(sub_layer)
        elif self.execution == 'sequential':
            order = self.w if self.sequential_order == 'ascending' else reversed(self.w)
            for axis in order:
                x = axis_call(x, self.w[axis], axis)
        return x

if __name__ == '__main__':
    import numpy as np
    import time
    # Define the spatial shape of the input tensor (excluding batch)
    shape = [16, 16, 16, 16]
    total_samples = 64
    batch_size_per_step = 32
    num_batches_per_epoch = total_samples // batch_size_per_step
    epochs = 16
    # Create dummy input and target data as PyTorch tensors
    # The input shape should be [total_samples] + shape
    x_torch = torch.randn(total_samples, *shape, dtype=torch.float32)
    y_torch = torch.randn(total_samples, *shape, dtype=torch.float32)
    # Instantiate the PyTorch MNN module
    # Using view='separate' and execution='parallel' as in the original example
    print("Instantiating MNN_torch with shape={}, view='separate', execution='parallel'".format(shape))
    model = MNN_torch(shape=shape, view='separate', execution='parallel')
    # Define Loss Function and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Print model structure and parameters (equivalent to model.summary() conceptually)
    print("\nPyTorch MNN Model Structure:")
    print(model)
    print("\nTrainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.shape}")
    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    # Training Loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        running_loss = 0.0
        # Process data in batches
        for i in range(num_batches_per_epoch):
            # Get batch data
            start_idx = i * batch_size_per_step
            end_idx = start_idx + batch_size_per_step
            batch_x = x_torch[start_idx:end_idx]
            batch_y = y_torch[start_idx:end_idx]
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(batch_x)
            # Calculate loss
            loss = criterion(outputs, batch_y)
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Print average loss for the epoch
        epoch_loss = running_loss / num_batches_per_epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    end_time = time.time()
    total_training_time = end_time - start_time
    print(f"\nTraining finished.")
    print(f"Total training duration: {total_training_time:.4f} seconds")
    # Optional: Evaluate the model after training (e.g., on the training data)
    # model.eval() # Set model to evaluation mode
    # with torch.no_grad():
    #     total_test_loss = 0.0
    #     # You might want to use a separate test dataset here
    #     test_outputs = model(x_torch)
    #     test_loss = criterion(test_outputs, y_torch)
    #     print(f"\nLoss on the full dataset after training: {test_loss.item():.4f}")


# jax implementation
import jax
import jax.numpy as jnp
import string
from flax import linen as nn
from typing import Union, List
import optax
from flax.training import train_state

class MNN_jax(nn.Module):
    shape: tuple
    view: Union[str, List[str]]
    execution: str = 'parallel'
    sequential_order: str = 'ascending'
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    def setup(self):
        # Process view configuration
        if isinstance(self.view, str):
            processed_view = [self.view.lower() for _ in self.shape]
        elif len(self.view) == 1:
            processed_view = [self.view[0].lower() for _ in self.shape]
        else:
            processed_view = [v.lower() for v in self.view]
        # Validate configuration
        if self.execution not in ['parallel', 'sequential']:
            raise ValueError("Execution must be 'parallel' or 'sequential'")
        if self.sequential_order not in ['ascending', 'descending']:
            raise ValueError("Sequential order must be 'ascending' or 'descending'")
        # Create parameters as attributes
        self.axes = list(range(-len(self.shape), 0))
        for axis in self.axes:
            view_type = processed_view[axis]
            if view_type == 'shared':
                in_shape = (self.shape[axis],)
            elif view_type == 'separate':
                in_shape = self.shape
            else:
                raise ValueError(f"Invalid view: {view_type}. Use 'shared' or 'separate'")
            param_shape = in_shape + (self.shape[axis],)
            setattr(self, f'w_{axis}', self.param(f'w_{axis}', self.kernel_init, param_shape))
    def axis_call(self, x, w, axis):
        equation_x = string.ascii_letters[:x.ndim]
        if len(w.shape) == 2:  # shared weights
            equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
        else:  # separate weights
            # Calculate starting index for equation_w
            start_idx = x.ndim - len(w.shape) + 1
            equation_w = equation_x[start_idx:] + string.ascii_letters[len(equation_x)]
        equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
        equation = f"{equation_x},{equation_w}->{equation_o}"
        return jnp.einsum(equation, x, w)
    def __call__(self, x):
        # Collect parameters in order
        params = [getattr(self, f'w_{axis}') for axis in self.axes]
        if self.execution == 'parallel':
            outputs = [axis_call(x, w, axis) for w, axis in zip(params, self.axes)]
            return sum(outputs)
        elif self.execution == 'sequential':
            order = zip(params, self.axes)
            if self.sequential_order == 'descending':
                order = reversed(list(order))
            for w, axis in order:
                x = axis_call(x, w, axis)
            return x


if __name__ == '__main__':
    import numpy as np
    import time
    # Example usage
    input_shape = (16, 16, 16, 16)
    batch_size = 64
    # Generate random data
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size,) + input_shape)
    y = jax.random.normal(key, (batch_size,) + input_shape)
    # Create model
    class Model(nn.Module):
        @nn.compact
        def __call__(self, x):
            return MNN_jax(shape=input_shape, view='separate', name='mnn')(x)
    # Initialize model
    model = Model()
    params = model.init(jax.random.PRNGKey(0), x)
    # Create optimizer and training state
    tx = optax.adam(0.001)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    # Define loss function
    def compute_loss(params, x, y):
        pred = model.apply(params, x)
        return jnp.mean((pred - y) ** 2)
    # Training step
    @jax.jit
    def train_step(state, batch_x, batch_y):
        grad_fn = jax.grad(lambda p: compute_loss(p, batch_x, batch_y))
        grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads)
    # Training loop
    t = time.time()
    for epoch in range(16):
        # For simplicity, we're using full-batch training here
        state = train_step(state, x, y)
        loss = compute_loss(state.params, x, y)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    t = time.time() - t
    print(t)

