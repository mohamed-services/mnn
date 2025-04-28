
import numpy as np

axes = 4
w = []
for i in range(axes):
    w.append(np.load(f'D:/mnn/model/weights/axis_{i}.npy'))

w0 = [np.copy(i) for i in w1]

for i in range(len(w0)): # find the axis to upscale
    axis = i
    if w0[0].shape[i] < w0[0].shape[i+1]:
        break

shape1 = np.array([i.shape for i in w0])
shape1[:,axis] *= 2
shape1[axis,-1] *= 2
shape1
w1 = [np.zeros(i) for i in shape1]
for i in range(len(w0)):
    if i != axis:
        w1[i] = np.concatenate([w0[i],w0[i]], axis=axis)
    else:
        ndim = w0[i].ndim
        shape_axis = w0[i].shape[axis]
        shape_last = w0[i].shape[-1]
        slice1 = [slice(None)] * ndim
        slice2 = [slice(None)] * ndim
        slice1[axis] = slice(None, shape_axis)
        slice2[axis] = slice(shape_axis, None)
        slice1[-1] = slice(None, shape_last)
        slice2[-1] = slice(shape_last, None)
        w1[i][tuple(slice1)] = w0[i]
        w1[i][tuple(slice2)] = w0[i]


for i in w1: i.shape


#for i in range(len(w)):
#    np.save(f'D:/mnn/model/weights/axis_{i}.npy', w[i])

