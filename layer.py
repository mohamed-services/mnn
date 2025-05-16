
def MNN(shape, 
        backend: str, 
        mode :str|list[str]='separate', 
        execution :str='parallel', 
        sequential_order: str='ascending', 
        single_axis :int|None=None, 
        axis_output :int|None=None, 
        kernel_initializer=None, 
        kernel_regularizer=None, 
        kernel_constraint=None, 
        weights :dict|None=None, 
        **kwargs):
    shape = list(shape) # shape of the input, must be a list of integers, doesn't include the batch size
    backend = backend.lower()
    if backend == 'tensorflow' or backend == 'keras' or backend == 'tf':
        import tensorflow as tf
        import keras
        layer = keras.layers.Layer(**kwargs)
        einsum = tf.einsum
    elif backend == 'torch' or backend == 'pytorch':
        import torch
        layer = torch.nn.Module(**kwargs)
        einsum = torch.einsum
    elif backend == 'jax' or backend == 'flax':
        import jax
        import flax
        layer = flax.linen.Module(**kwargs)
        einsum = jax.numpy.einsum
    else:
        raise ValueError('backend value must be tensorflow or torch or jax')
    execution = execution.lower()
    if execution not in ['parallel', 'sequential', 'single']:
        raise ValueError('execution value must be parallel or sequential or single')
    if execution == 'single':
        if single_axis is None or axis_output is None or single_axis >= len(shape) or single_axis < -len(shape):
            raise ValueError('single_axis and axis_output must be provided and valid')
    if type(mode) == str:
        mode = [mode.lower() for _ in shape]
    elif len(mode) == 1:
        mode = [mode[0].lower() for _ in shape]
    if len(mode) != len(shape):
        raise ValueError(f"Length of mode list ({len(mode)}) must match length of shape ({len(shape)}).")
    if sequential_order.lower() == 'ascending':
        axes = range(-len(shape),0) 
    elif sequential_order.lower() == 'descending':
        axes = reversed(range(-len(shape),0))
    else:
        raise ValueError('sequential_order value must be ascending or descending')
    if execution == 'single':
        axes = [min(single_axis, single_axis-len(shape))]
    w = {}
    if weights:
        w = weights
        axes = []
    for axis in axes:
        if weights: break
        if mode[axis] == 'shared':
            in_shape = [shape[axis]]
        elif mode[axis] == 'separate':
            in_shape = list(shape)
        else:
            raise ValueError('mode value invalid, valid mode values are shared or separate or mixture list of both')
        out_shape = [shape[axis]]
        if execution == 'single':
            out_shape = [axis_output]
        if backend == 'tensorflow' or backend == 'keras' or backend == 'tf':
            w[axis] = layer.add_weight(shape=in_shape+out_shape, initializer=kernel_initializer, regularizer=kernel_regularizer, constraint=kernel_constraint)
        elif backend == 'torch' or backend == 'pytorch':
            param = torch.nn.Parameter(torch.empty(in_shape+out_shape))
            torch.nn.init.xavier_uniform(param)
            param_name = f'weight_{axis}'
            layer.register_parameter(param_name, param)
            w[axis] = param
        elif backend == 'jax' or backend == 'flax':
            w[axis] = layer.param(f'weight_{axis}', shape=in_shape+out_shape, init_fn=flax.linen.initializers.glorot_uniform)
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
        return einsum(equation, x, w)
    def MNN_call(x):
        if execution == 'parallel':
            sub_layer = []
            for axis in w:
                sub_layer.append(axis_call(x, w[axis], axis))
            x = sum(sub_layer)
        elif execution == 'sequential' or execution == 'single':
            for axis in w:
                x = axis_call(x, w[axis], axis)
        else:
            raise ValueError('execution value must be parallel or sequential or single')
        return x
    if backend == 'tensorflow' or backend == 'keras' or backend == 'tf':
        layer.call = MNN_call
    elif backend == 'torch' or backend == 'pytorch':
        layer.forward = MNN_call
    elif backend == 'jax' or backend == 'flax':
        layer.__call__ = MNN_call
    return layer

