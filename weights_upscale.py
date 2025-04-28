
import numpy as np

w0 = np.load('weights.npy')

# find the axis to upscale
for i in range(len(w0)):
    axis = i
    if w0[0].shape[i] < w0[0].shape[i+1]:
        break

shape1 = np.array([i.shape for i in w0])
shape1[:,axis] *= 2
shape1[axis,-1] *= 2
shape1
w1 = [np.zeros(i, dtype=w0.dtype) for i in shape1]
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

if axis == len(w0)-1:
    # upscale the embeddings and the unembeddings
    pass

#save and test w1

