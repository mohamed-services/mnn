
# tensorflow implementation
import tensorflow as tf
import keras
import string

def axis_call(x, w, axis):
    equation_x = string.ascii_letters[:len(x.shape)]
    if len(w.shape) == 2: #shared
        equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
    else: #separate
        equation_w = equation_x[1-len(w.shape):] + string.ascii_letters[len(equation_x)]
    equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
    equation = equation_x + ',' + equation_w + '->' + equation_o
    print(equation)
    return tf.einsum(equation, x, w)

class MNN(keras.layers.Layer):
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

class resizing_layer(keras.layers.Layer):
    def __init__(self, shape, axis: int, output_shape: int, sharing: bool, **kwargs):
        super().__init__()
        self.shape = shape
        self.axis = axis
        self.w = {}
        if sharing == True:
            self.w[axis] = self.add_weight(shape=[shape[axis], output_shape])
        elif sharing == False:
            self.w[axis] = self.add_weight(shape=list(shape) + [output_shape])
    def call(self, x):
        x = axis_call(x, self.w[self.axis], self.axis)
        return x

# add preinitialized weights in the init


# torch implementation
import torch
import string

def axis_call(x, w, axis):
    equation_x = string.ascii_letters[:len(x.shape)]
    if len(w.shape) == 2: #shared
        equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
    else: #separate
        equation_w = equation_x[1-len(w.shape):] + string.ascii_letters[len(equation_x)]
    equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
    equation = equation_x + ',' + equation_w + '->' + equation_o
    #print(equation)
    return torch.einsum(equation, x, w)

class MNN(torch.nn.Module):
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

class resizing_layer(keras.layers.Layer):
    def __init__(self, shape, axis: int, output_shape: int, sharing: bool, **kwargs):
        super().__init__()
        self.shape = shape
        self.axis = axis
        self.w = {}
        if sharing == True:
            self.w[axis] = self.add_weight(shape=[shape[axis], output_shape])
        elif sharing == False:
            self.w[axis] = self.add_weight(shape=list(shape) + [output_shape])
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
        MNN(shape, 'separate', )
    ])
    model.compile('adam', 'mse')
    model.summary()
    t = time.time()
    model.fit(x, y, epochs=16)
    t = time.time() - t
    print(t)

    

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
    model = MNN(shape=shape, view='separate', execution='parallel')
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

