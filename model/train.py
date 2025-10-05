
import numpy as np
import os
import tensorflow as tf
import keras
from mnn.layer import MNN # type: ignore

dims = 4
path = 'D:/mnn/model/weights/'
w = []
for i in range(dims):
    w.append(np.load(path+f'axis_{i}.npy'))

axes = list(range(-dims,0)) 
shape = [i.shape[-1] for i in w]
shape
weights = {axes[i]:w[i] for i in range(dims)}

inputs = keras.layers.Input(shape)
layer = inputs
mnn_layer = MNN(shape, backend='tf', weights=weights)
relu = keras.layers.ReLU()

for i in range(7):
    layer = mnn_layer(layer)
    layer = relu(layer)

layer = mnn_layer(layer)

model = keras.Model(inputs, layer)

model.compile('adam', 'mse')

model.summary()

def binary(size):
    return np.array([[int(j) for j in bin(i)[2:].zfill(size)] for i in range(2**size)])

size = 8
x = binary(size) 
x = x * 2 - 1
x = np.pad(x, [[0,0],[0,np.prod(shape)-size]])
x = np.concatenate([np.roll(x, i, axis=1) for i in range(16, np.prod(shape), 8)], axis=0)
x.shape = [x.shape[0], *shape]
x.shape

model.evaluate(x, x)

model.fit(x, x, batch_size=x.shape[0], epochs=2**16)

'''
w = model.get_weights()
for i in range(len(w)):
    np.save(path+f'axis_{i}.npy', w[i])

'''
