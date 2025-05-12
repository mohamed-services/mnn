
import numpy as np
import string
import os
import itertools
import tensorflow as tf
import keras
from mnn.layer import MNN  # type: ignore

path = 'D:/mnn/model/sgd/weights/'
w = []
for i in range(len(os.listdir(path))):
    w.append(np.load(path+f'axis_{i}.npy').astype(np.float64))

shape = [i.shape[-1] for i in w]
shape

class activate(keras.layers.Layer):
    def call(self, x):
        shape = tf.shape(x)
        rand = tf.cast(tf.round(tf.random.uniform(shape=[], dtype=tf.float32)) * 2 - 1, 'float32')
        x = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [tf.shape(x)[-1] // 2, 2]], axis=0))
        x = tf.split(x, 2, -1)
        a = [x[0] + x[1], tf.abs(x[0] - x[1])]
        half = tf.constant([0.5], dtype='float32')
        o0 = ( a[0]) + ( a[1]) 
        print(o0.shape)
        o1 = a[0] - a[1]
        print(o1.shape)
        o = tf.concat([o0, o1], axis=-1)
        return tf.reshape(x, shape)

inputs = keras.layers.Input(shape)
layer = MNN(shape, 'tf')(inputs)

model = keras.Model(inputs, layer)

model.compile(keras.optimizers.SGD(2.0**-4), 'mse')

model.summary()

model.set_weights(w)

def binary(size):
    return np.array([[int(j) for j in bin(i)[2:].zfill(size)] for i in range(2**size)])

size = 2
x = binary(size) * 2 - 1
x
x.shape = [x.shape[0], *shape]

model.evaluate(x, x)

model.fit(x,x, epochs=2**8)
