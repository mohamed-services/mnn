
import numpy as np
import string
import os
import itertools
import tensorflow as tf
import keras
from mnn.layer import MNN  # type: ignore

path = 'D:/mnn/model/gd/weights/'
w = []
for i in range(len(os.listdir(path))):
    w.append(np.load(path+f'axis_{i}.npy').astype(np.float64))

shape = [i.shape[-1] for i in w]
shape

inputs = keras.layers.Input(shape)
layer = MNN(shape, 'tf')(inputs)

model = keras.Model(inputs, layer)

model.compile(keras.optimizers.SGD(2**-4), 'mse')

model.summary()

