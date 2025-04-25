
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
        self.execution = execution
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
            x = tf.add_n(sub_layer)
        elif self.execution == 'sequential':
            order = self.w if self.sequential_order == 'ascending' else reversed(self.w)
            for axis in order:
                x = axis_call(x, self.w[axis], axis)
        else:
            raise ValueError('execution value must be parallel or sequential')
        return x

class resizing_layer(keras.layers.Layer):
    def __init__(self, shape, axis: int, output_shape: int, spatial_sharing: bool, **kwargs):
        super().__init__()
        self.shape = shape
        self.axis = axis
        self.w = {}
        if spatial_sharing == True:
            self.w[axis] = self.add_weight(shape=[shape[axis], output_shape])
        elif spatial_sharing == False:
            self.w[axis] = self.add_weight(shape=list(shape) + [output_shape])
    def call(self, x):
        x = axis_call(x, self.w[self.axis], self.axis)
        return x

