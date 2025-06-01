
from string import ascii_letters
import types

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

def get_base_layer(backend, w_shapes, kernel_initializer, weights, **kwargs):
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
        for axis, w_shape in w_shapes:
            if kernel_initializer == None:
                kernel_initializer = keras.initializers.RandomUniform(minval=-w_shape[-1]**-0.5, maxval=w_shape[-1]**-0.5)
            if weights:
                kernel_initializer = keras.initializers.Constant(weights[axis])
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
        for axis, w_shape in w_shapes:
            if weights:
                param = torch.nn.Parameter(torch.from_numpy(weights[axis]))
            else:
                param = torch.nn.Parameter(torch.empty(w_shape))
                torch.nn.init.uniform(param, a=-w_shape[-1]**-0.5, b=w_shape[-1]**-0.5)
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
        for axis, w_shape in w_shapes:
            current_init_fn = nn.initializers.uniform(w_shape[-1]**-0.5) 
            if weights:
                pre_trained_weight = weights[axis]
                current_init_fn = lambda _key, _shape, _dtype: jnp.asarray(pre_trained_weight, dtype=_dtype)
            layer.w[axis] = layer.param(f'weight_{axis}', current_init_fn, w_shape)
        return layer
    else:
        raise ValueError(
            f"Unsupported backend: '{backend}'. Supported backends are: {', '.join(list(backend_aliases.keys()))}."
        )

def init(shape, # shape of the input, must be a list of integers, doesn't include the batch size
        backend: str, 
        mode :str|list[str]='separate', 
        execution_order :str='parallel', 
        sequential_order: str='ascending', 
        single_axis :int|None=None, 
        axis_output :int|None=None, 
        kernel_initializer=None, 
        weights :dict|None=None, 
        **kwargs):
    # validate shape input value is a list 
    shape = list(shape) 
    # validate mode input value
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
        w_shapes.append([axis, in_shape+out_shape])
    # validate backend input value and create the layer instance 
    layer = get_base_layer(backend, w_shapes, kernel_initializer, weights, **kwargs)
    layer.execution_order = execution_order
    return layer

