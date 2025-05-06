
import numpy as np
import string
import os
import itertools
import tensorflow as tf
import keras
import D:/mnn/layer.py 

path = 'D:/mnn/model/gd/weights/'
w = []
for i in range(len(os.listdir(path))):
    w.append(np.load(f'D:/mnn/model/ea/weights/axis_{i}.npy').astype(np.float64))

shape = [i.shape[-1] for i in w]
shape

