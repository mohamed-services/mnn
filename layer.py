
import string
from string import ascii_letters
import types
from typing import Iterable
import sys

def axis_call(x, w, axis, einsum):
    equation_x = ascii_letters[:len(x.shape)]
    if len(w.shape) == 2: #shared
        equation_w = equation_x[axis] 
    else: #separate
        equation_w = equation_x[1-len(w.shape):] 
    equation_w += ascii_letters[len(equation_x)]
    equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
    equation = equation_x + ',' + equation_w + '->' + equation_o
    #print(equation)
    return einsum(equation, x, w)

def layer_call(self, x):
    if self.execution_order == 'parallel':
        sub_layer = []
        for axis in self.w:
            sub_layer.append(axis_call(x, self.w[axis], axis, self.einsum))
        x = sum(sub_layer)
    else: # execution_order == 'sequential' or execution_order == 'single'
        for axis in self.w:
            x = axis_call(x, self.w[axis], axis, self.einsum)
    return x

def get_base_layer(backend, w_shapes, weights, **kwargs):
    """Dynamically imports and returns backend-specific base layer and einsum function."""
    backend_aliases = {
    'tensorflow': ['tensorflow', 'keras', 'tf'],
    'pytorch': ['torch', 'pytorch'],
    'jax': ['jax', 'flax'],
    }
    backend = backend.lower()
    if backend in backend_aliases['tensorflow']:
        try:
            import tensorflow as tf
            import keras
        except ImportError:
            raise ImportError("TensorFlow/Keras backend selected, but not installed.")
        layer = keras.layers.Layer(**kwargs)
        layer.call = types.MethodType(layer_call, layer)
        layer.einsum = tf.einsum
        layer.backend = 'tensorflow'
        layer.w = {}
        for axis in w_shapes:
            w_shape = w_shapes[axis]
            if weights:
                kernel_initializer = keras.initializers.Constant(weights[axis])
            else:
                kernel_initializer = keras.initializers.RandomUniform(minval=-sum(w_shape[:-1])**-0.5, maxval=sum(w_shape[:-1])**-0.5)
            layer.w[axis] = layer.add_weight(shape=w_shape, initializer=kernel_initializer, trainable=True, name=f'weight_{axis}')
        return layer
    elif backend in backend_aliases['pytorch']:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch backend selected, but not installed.")
        layer = torch.nn.Module(**kwargs)
        layer.forward = types.MethodType(layer_call, layer)
        layer.einsum = torch.einsum
        layer.backend = 'pytorch'
        layer.w = {}
        for axis in w_shapes:
            w_shape = w_shapes[axis]
            if weights:
                param = torch.nn.Parameter(torch.from_numpy(weights[axis]))
            else:
                param = torch.nn.Parameter(torch.empty(w_shape))
                torch.nn.init.uniform(param, a=-sum(w_shape[:-1])**-0.5, b=sum(w_shape[:-1])**-0.5)
            layer.register_parameter(f'weight_{axis}', param)
            layer.w[axis] = param
        return layer
    elif backend in backend_aliases['jax']:
        try:
            import jax
            import flax.linen as nn
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX/Flax backend selected, but not installed.")
        layer = nn.Module(**kwargs)
        layer.__call__ = types.MethodType(layer_call, layer)
        layer.einsum = jax.numpy.einsum
        layer.backend = 'jax'
        layer.w = {}
        for axis in w_shapes:
            w_shape = w_shapes[axis]
            if weights:
                init_fn = nn.initializers.constant(weights[axis])
            else:
                init_fn = nn.initializers.uniform(sum(w_shape[:-1])**-0.5) 
            layer.w[axis] = layer.param(f'weight_{axis}', init_fn, w_shape)
        return layer
    else:
        raise ValueError(f"Unsupported backend: '{backend}'. Supported backends are: {', '.join(list(backend_aliases.keys()))}.")

def get_mode(shape, mode, execution_order, sequential_order, single_axis, axis_output):
    if type(mode) == str:
        mode = [mode.lower() for _ in shape]
    elif len(list(mode)) == 1:
        mode = [mode[0].lower() for _ in shape]
    elif len(list(mode)) != len(shape):
        raise ValueError(f"Length of mode list ({len(mode)}) must match length of shape ({len(shape)}).")
    if sequential_order.lower() == 'ascending':
        axes = range(-len(shape),0) 
    elif sequential_order.lower() == 'descending':
        axes = reversed(range(-len(shape),0))
    else:
        raise ValueError('sequential_order value must be ascending or descending')
    # validate execution input value
    execution_order = execution_order.lower()
    if execution_order not in ['parallel', 'sequential', 'single']:
        raise ValueError('execution_order value must be parallel or sequential or single')
    if execution_order == 'single':
        if single_axis is None or axis_output is None or single_axis >= len(shape) or single_axis < -len(shape):
            raise ValueError('single_axis and axis_output must be provided and valid')
        else:
            axes = [min(single_axis, single_axis-len(shape))]
    return axes, mode, execution_order

def __init__(shape: list|tuple|Iterable, # shape of the input, must be a list of integers, doesn't include the batch size
        backend: str, 
        mode :str|list[str]='separate', 
        execution_order :str='parallel', 
        sequential_order: str='ascending', 
        single_axis :int|None=None, 
        axis_output :int|None=None, 
        weights :dict|None=None, 
        **kwargs):
    # validate shape input value is a list 
    try:
        shape = list(shape) 
    except Exception as e:
        raise ValueError(f"shape must be a list or tuple or iterable, got {type(shape)}") from e
    # validate mode input value
    axes, mode, execution_order = get_mode(shape, mode, execution_order, sequential_order, single_axis, axis_output)
    # calculate the weights shapes for each axis
    w_shapes = {}
    for axis in axes:
        if mode[axis] == 'shared':
            in_shape = [shape[axis]]
        elif mode[axis] == 'separate':
            in_shape = list(shape)
        else:
            raise ValueError('mode value invalid, valid mode values are shared or separate or mixture list of both')
        out_shape = [shape[axis]]
        if execution_order == 'single':
            out_shape = [axis_output]
        w_shapes[axis] = in_shape+out_shape
    # validate backend input value and create the layer instance 
    layer = get_base_layer(backend, w_shapes, weights, **kwargs)
    # set execution order 
    layer.execution_order = execution_order
    return layer





# tensorflow implementation
# import tensorflow and keras if they are available
# you must import the backend before importing this module
if 'tensorflow' in sys.modules and 'keras' in sys.modules:
    import tensorflow as tf
    import keras

class MNN_tf(keras.layers.Layer):
    def __init__(self, shape, mode :str|list[str]='separate', execution :str='parallel', sequential_order: str='ascending', single_axis :int|None=None, axis_output :int|None=None, kernel_initializer=None, kernel_regularizer=None, kernel_constraint=None, weights=None, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape # shape of the input, must be a list of integers, doesn't include the batch size
        self.execution = execution.lower()
        if self.execution not in ['parallel', 'sequential', 'single']:
            raise ValueError('execution value must be parallel or sequential or single')
        if weights:
            self.w = weights
            return
        if self.execution == 'single':
            if single_axis is None or axis_output is None or single_axis >= len(shape) or single_axis < -len(shape):
                raise ValueError('single_axis and axis_output must be provided and valid')
        if type(mode) == str:
            mode = [mode.lower() for i in shape]
        elif len(mode) == 1:
            mode = [mode[0].lower() for i in shape]
        if len(mode) != len(shape):
            raise ValueError(f"Length of mode list ({len(mode)}) must match length of shape ({len(self.shape)}).")
        if sequential_order.lower() == 'ascending':
            axes = range(-len(self.shape),0) 
        elif sequential_order.lower() == 'descending':
            axes = reversed(range(-len(self.shape),0))
        else:
            raise ValueError('sequential_order value must be ascending or descending')
        if self.execution == 'single':
            axes = [min(single_axis, single_axis-len(shape))]
        self.w = {}
        for axis in axes:
            if mode[axis] == 'shared':
                in_shape = [shape[axis]]
            elif mode[axis] == 'separate':
                in_shape = list(shape)
            else:
                raise ValueError('mode value invalid, valid mode values are shared or separate or mixture list of both')
            if self.execution == 'single':
                self.w[axis] = self.add_weight(shape=in_shape+[axis_output], initializer=kernel_initializer, regularizer=kernel_regularizer, constraint=kernel_constraint)
                break
            self.w[axis] = self.add_weight(shape=in_shape+[shape[axis]], initializer=kernel_initializer, regularizer=kernel_regularizer, constraint=kernel_constraint)
    def axis_call(self, x, w, axis):
        equation_x = string.ascii_letters[:len(x.shape)]
        if len(w.shape) == 2: #shared
            equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
        else: #separate
            equation_w = equation_x[1-len(w.shape):] + string.ascii_letters[len(equation_x)]
        equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
        equation = equation_x + ',' + equation_w + '->' + equation_o
        #print(equation)
        return tf.einsum(equation, x, w)
    def call(self, x):
        if self.execution == 'parallel':
            sub_layer = []
            for axis in self.w:
                sub_layer.append(self.axis_call(x, self.w[axis], axis))
            x = sum(sub_layer)
        elif self.execution == 'sequential' or self.execution == 'single':
            for axis in self.w:
                x = self.axis_call(x, self.w[axis], axis)
        else:
            raise ValueError('execution value must be parallel or sequential or single')
        return x


# torch implementation
# import torch if it is available
# you must import the backend before importing this module
if 'torch' in sys.modules:
    import torch

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
                sub_layer.append(self.axis_call(x, self.w[axis], axis))
            x = sum(sub_layer)
        elif self.execution == 'sequential':
            order = self.w if self.sequential_order == 'ascending' else reversed(self.w)
            for axis in order:
                x = self.axis_call(x, self.w[axis], axis)
        return x


# jax implementation
# import jax and flax if they are available
# you must import the backend before importing this module
if 'jax' in sys.modules and 'flax' in sys.modules:
    import jax.numpy as jnp
    from flax import linen as nn
    from typing import Union, List

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
            outputs = [self.axis_call(x, w, axis) for w, axis in zip(params, self.axes)]
            return sum(outputs)
        elif self.execution == 'sequential':
            order = zip(params, self.axes)
            if self.sequential_order == 'descending':
                order = reversed(list(order))
            for w, axis in order:
                x = self.axis_call(x, w, axis)
            return x

