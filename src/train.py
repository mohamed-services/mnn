
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
mnn_layer = MNN(shape, backend='tf', weights=weights)
relu = keras.layers.ReLU()
layer = inputs

for i in range(7):
    layer = mnn_layer(layer)
    layer = relu(layer)

layer = mnn_layer(layer)

mask_inputs = keras.layers.Input(shape)
layer = keras.layers.Multiply()([layer, mask_inputs])

model = keras.Model([inputs, mask_inputs], layer)

def err_max(y_true, y_pred):
    return tf.reduce_max(tf.abs(tf.cast(y_true, dtype=tf.float32) - y_pred))

def err_sum(y_true, y_pred):
    return tf.reduce_sum(tf.abs(tf.cast(y_true, dtype=tf.float32) - y_pred))

model.compile(keras.optimizers.Adam(learning_rate=2**-10), 'mse', metrics=[err_sum, err_max])
#model.compile(keras.optimizers.Adam(learning_rate=2**-10, weight_decay=2**-10), 'mse', metrics=[err_sum, err_max])
#model.compile(keras.optimizers.SGD(learning_rate=2**-4, weight_decay=2**-4), 'mse', metrics=[err_sum, err_max])

model.summary()

def binary(size):
    return np.array([[int(j) for j in bin(i)[2:].zfill(size)] for i in range(2**size)])


size = 8
x = binary(size) 
x = x * 2 - 1
mask = np.ones(x.shape)
mask = np.pad(mask, [[0,0],[0,np.prod(shape)-size]])
mask = np.concatenate([np.roll(mask, i, axis=1) for i in range(0, np.prod(shape), 8)], axis=0)
mask.shape = [mask.shape[0], *shape]
x = np.pad(x, [[0,0],[0,np.prod(shape)-size]])
x = np.concatenate([np.roll(x, i, axis=1) for i in range(0, np.prod(shape), 8)], axis=0)
x.shape = [x.shape[0], *shape]
x.shape


'''
dataset_file_name = 'D:/wiki/alphabet.txt'
#dataset_file_name = 'D:/wiki/vocab.txt'
with open(dataset_file_name, 'r', encoding='utf-8') as f:
    text = [list(i.encode('utf-8')) for i in f.read()]

'''

model.evaluate([x, mask], x, batch_size=x.shape[0])

model.fit([x, mask], x, epochs=2**8)
model.fit([x, mask], x, batch_size=x.shape[0], epochs=2**16)

'''
w = model.get_weights()
for i in range(len(w)):
    np.save(path+f'axis_{i}.npy', w[i])

'''


np.abs(w[0]).max()
np.abs(w[1]).max()
np.abs(w[2]).max()
np.abs(w[3]).max()
np.abs(w[0]).mean()
np.abs(w[1]).mean()
np.abs(w[2]).mean()
np.abs(w[3]).mean()
w = model.get_weights()
np.abs(w[0]).max()
np.abs(w[1]).max()
np.abs(w[2]).max()
np.abs(w[3]).max()
np.abs(w[0]).mean()
np.abs(w[1]).mean()
np.abs(w[2]).mean()
np.abs(w[3]).mean()
