
import numpy as np
import os
import tensorflow as tf
import keras
from mnn.layer import MNN # type: ignore

path = 'D:/mnn/model/weights/'
w = []
for i in range(len(os.listdir(path))):
    w.append(np.load(path+f'axis_{i}.npy').astype(np.float32))

shape = [i.shape[-1] for i in w]
shape

inputs = keras.layers.Input(shape)
layer = inputs
mnn_layer = MNN(shape, 'tf')
relu = keras.layers.ReLU()

layer = mnn_layer(layer)
layer = relu(layer)
layer = mnn_layer(layer)

model = keras.Model(inputs, layer)

model.set_weights(w)

model.compile(keras.optimizers.SGD(2**-2), 'mse')

model.summary()

def binary(size):
    return np.array([[int(j) for j in bin(i)[2:].zfill(size)] for i in range(2**size)])

size = 8
x = binary(size) * 2 - 1
x
x.shape = [x.shape[0], *shape]

model.evaluate(x, x)

model.fit(x,x, epochs=2**16)

'''
w = model.get_weights()
for i in range(len(w)):
    np.save(path+f'axis_{i}.npy', w[i])

'''
