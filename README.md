# Multidimensional neural networks
  
Mohamed Ibrahim
  
## Abstract

This is an open source paper so anyone is more than welcome to clone it, improve upon it and send a pull request to modify it.
  
## Introduction

I feel that I need to introduce myself to clear the motivation and to set some expectations, programming and machine learning was a little hobby of mine but I gave up on this hobby some years ago, and I’m trying to get back to it right now because I need an AI model to be my assistant and the large companies don't provide what I exactly need, there were a lot of advancements in the field in the past few years so I'm trying to catch up.  
The purpose of this paper is not to prove point/s nor to introduce some new discovery/ies, the purpose of this paper is to help you create a full general purpose AI model on your personal computer from scratch and scale it based on your requirements whatever it is and get similar results to some of the popular models.  
I know that there’re some open source LLM models on the web but most of the open source models that can be downloaded and run locally is very inefficient in regards to the resources usage and non of them is general purpose and we need a general purpose model not just an LLM, we need a model that can handle multiple modalities at the same time like text, audio, and video and has multiples modes like autoregressive and diffusion and can alternate between factual, creative, and reasoning easily and can use tools and surf the web and to be able to act as a human employee or a human assistant so it needs to be able to run continuously for many hours, days, weeks, and months, which is not possible using current architucures like trasnformers so I have to create a new architucure and model for myself and with plenty of time with nothing to do, I decided to document and to share the process with you while working on this project.  
So to start we need first a good architecture.
Usually training such a model takes millions if not billions of dollars in resources, So reducing the costs like a thousand fold to be able to accomplish it using a PC is not an easy task, I know that I need to move from dynamical architectures like transformers to a fixed architecture like MLP with dynamical resources assignment inspired by the biological brains, and from two dimensions to higher dimensions, and from full training to partial training and blind training, and from finetuning to identity embedding, and maybe all of this will not be enough, I know that every step in the process needs to be optimized or completely changed to achieve that, and there’re a lot of things to work on and it will take a long time, but I know that eventually we will get the wanted results.  
Also be cautious about any mathematical calculation, code snippet or any unproved claim in this paper, as maybe further verification is needed.
  
## Multidimensionality

If we are using one dimensional neural network like MLP architecture for our model then, if we have an image of size (600, 600, 3) then its flatten context window is approximately 1 million token then we need 1 trillion parameters for a single layer to process this image, also if we are dealing with a text of size 16k words and embedded every word in a vector of size 64 so its flatten context window will be approximately 1 million token then  we need 1 trillion parameters per layer to process this text, and the truth is most of those 1 trillion parameters doesn’t contribute much and we can convert a lot of those parameters to zeros without affecting the final results much, So the problem of MLP is that it scales quadratically that’s why we are not using  MLP for tasks of large size like images or text,
The transformer models decrease this problem by reshaping the flat inputs like text into 2d arrays for example, from 1 million to (16384, 64) for that text input, and cut and reshape the image into patches for example from 1 million to (5625, 192), so the computational complexity is reduced from 1 million squared to  (32768^2, 32^2) and (5625^2, 192^2),  The logical conclusion is that we should not stop at just two dimensions as increasing the dimensions reduces the needed computation, so we can reduce the computation by reshaping the inputs from two dimensions to three or more, if we want we can reshape the inputs to a maximum of dimensions equals  log_2 of its size, so every dimension will be of size two,
The transformer model can process the inputs in multidimensional space, for example if you have input sample of ten thousand words and embedding of size 64 usually you can run the attention mechanism on the first axis of size 10,000 and then you will run the feedforward layer on the second axis of 64, or you can reshape this sample to be (100, 100, 64) and run the attention mechanism on the first axis of size 100 then run the attention mechanism on the second axis of size 100 and then run the feedforward layer on the third axis of size 64, another example if you have an input of size one million words then you can run the attention mechanism on the first axis of size 1,000,000 and run the feedforward layer on the second axis of size 64, or you can reshape it to (100, 100, 100, 64) and then run the attention mechanism of size 100 on the first axis then run it on the second axis of size 100 then run it on the third axis of size 100 then run the feedforward layer on the fourth axis of size 64, I’ll not use the attention mechanism in my model, I’m mentioning it just in case you want to use it,
You can implement multidimensional layers using attention layers or convolutional layers, or locally connected layers or feedforward layers or many other options.
  
## Feedforward neural networks

Feedforward layers, is a static or fixed size type of layers where you must decide the input size and the output size of the layer before creating it, which is different from the dynamic range that the attention mechanism gives us, theoretically you can give the transformer model an insanely large input in size and it will compute it and return you the output, but realistically most LLM platforms and transformer models out there have limits to the input size that you can feed to the model and the output size that you can get from the model, right now most of the models have limited context window that’s less than one million words, and some below hundred thousand words,  and as soon as you decide the maximum size for your model whatever it is, then the attention mechanism isn't needed anymore and you can just use feedforward layer instead of it and zero pad your inputs to the needed size, so instead of using the attention mechanism on the first axis and the feedforward layer on the second axis, you can just use feedforward layer on the first axis and another feedforward layer on the second axis and you will get the same outputs, also you can make the feedforward layer sparse or more dynamic by deciding how many nodes you want to be used for any specific input out of the total number of nodes in that layer based on input size or the task difficulty,
  
## Multidimensional layers

Multidimensional neural network, is like any ordinary feedforward neural network,  
If we have one fully connected layer of one hundred neurons with input size of one hundred then this fully connected layer will have ten thousand connections (100^2), We can reduce the number of parameters while keeping this layer fully connected by reshaping the input from one dimension of one hundred to two dimensions of ten by ten and run ten different dense layers with ten neurons inside on each row and run another ten dense layers with ten neurons inside on each column and sum the two sets, then the number of the parameters will be (10^3)+(10^3) which is equal to 2,000 parameters but because we converted the 1d input to 2d then we need to run the sublayers on the two axes sequentially over two steps or we will need to stack two layers, because the maximum number of steps needed to move information from any point to any point using a straight line in a multidimensional array is equal to the number of dimensions so we need one step for 1d and two steps for 2d and three steps for 3d and so on,
And those numbers are for non-shared or spatially separate parameters, but if we will share the parameters inside every axis the number of parameters will be ((10^2) + (10^2)) = 200 shared parameters, also if all the axes are of the same size then we can share the parameters of one axis with all the axes and the number of parameters will be (10^2) = 100, and in general you can decide what parameters you want to be shared inside your model and how it will be shared between the neurons and what parameters you don’t want to be shared inside your model based on your requirements,
  
## Layer resizing

The multidimensional layers give you the freedom to resize the layers shape whatever you want by using a dense layer on the specific axis you want to resize, like how the feedforward layers allows you to get different output size than the input,  

## Biases

Note that the biases are not accounted for in the above calculations because if you will use an embedding layer as an input layer then using the biases is optional inside the hidden layers, but you’re free to use biases in your model,
  
## Activation function

The following activation functions are the candidates for the mnn model  
relu: pros (simple to compute, sparse, recommended overall) cons (sharp, dying nodes)  
leakyrelu: leakyrelu(x) = max(x, 0.25*x)  
hardtanh: hardtanh(x) = clip(x, -1, 1). recomended if you will quantize your model and will use fixed point arithmetic as it more compatible with a Fixed-point arithmetic by making the minimum and the maximum limit of the datatype the same as the activation function for example an 8 bit datatype will have 256 values from approximatly negative one to approximatly positive one and any lower or higher value will be clipped naturally by the datatype  
elu: elu(x) = x if x >= 0 else exp(x) - 1  
melu: melu(x) = x*exp(min(x,0)). melu is a variant of activation functions like gelu, silu, and mish but it is designed for deeper networks as it have better data flow compared to gelu, silu, and mish  
sort2: sort2(x) = reshape(x, [-1, 2]); sort(x, axis=-1); reshape(x, original_shape)  

Some models may perform better with a partial activation instead of full activation, I think you should give it a try, for example you can apply the activation function on the first three quarters of the nodes and leave the remaining quarter as linear as it is, because the network needs linear and non-linear information to be passed from layer to layer, so by keeping some outputs linear you improve the forward and backward data flow in your model,
  
## Connectivity

There're a lot of ways to wire a multidimensional layer for example you can make it fully wired or wire it randomly or wire nodes to their close neighbors using sliding n-dimensional convolution kernels, or using neural architecture search to find the best wiring pattern, you can use any dense or sparse wiring technique that suits your targets, I’ll will wire my model using axis based point of view like using rows and columns etc., so every node will be connected to every other node that shares all the dimensional coordinates or indexes with it but doesn’t share exactly one coordinate or index with it as explained above,
  
## Network design

you're free to design the network the way you want by choosing how many dimensions and the size of every dimension also the number of the hidden layers and what kind of layers to include and what not to include so you're not bounded by any design pattern, and for my model I’ll use one multidimensional layer with separate or non-shared parameters , I’ll start with a small context window and will scale the model up slowly while training, also the previous outputs will be refed to the network multiple times before getting the final outputs, I know that I should use shared parameters architecture with lower number of dimensions and large dimension size and multiple hidden layers and that is the right way to do it, but I'll use fully separate parameters architecture for my model because it will be more suitable for the training techniques that I’ll use later,  
  
A fun side note you can simulate a human brain with a layer of shape (4096, 4096, 4096) with action potentials and STDP learning in real time on an exaflop server that costs less than 500 million dollars, that’s just to show that our problem is more in the software like the designs and the architectures that we are using than in the hardware, if the problem was in the hardware then the world have now the needed hardware to run an artificial super intelligence system, and within two or three decades the hardware will get cheaper till every child will have enough hardware to download and run an artificial super intelligence system on his computer,  
I’m not saying that I have the designs and architecture needed to do it, I’m pointing out how much we need to work on our current methods, I’m in a position right now that I don’t have the hardware nor the designs and architectures to build a simple general-purpose model,  

## Tokenization

I'll use bit based tokenization for all modalities, So every token will equal one bit of two binary values negative one and psoitive one, So the UTF8 text may take from 8 to 32 bits, Images signals will take 24 bits, Audio signals will take 16 bits,
I'll not use any embedding nor unembedding layers and the model will handle the inputs and outputs binary values directly, Not using embedding and unembedding layers isn't the recommended way for the tokenization but it's more suitable for my constrained hardware,  

## Padding

Transformer models gives flexible context window, but the feedforward models needs fixed context window so we have to pad our inputs to a fixed size so the feedforward layers can process it, you can leave the beginning of your inputs and only pad the end of the inputs, but for me all the inputs will be padded with random number of zeros before it and the inputs will be zero padded after it till the end of the model input size so the position of the input will be random not in the beginning of the context window,  
  
## Context window

I’ll start my model with a context window of one character or one pixel then I’ll start to scale it up and increase the context window,
My final goal will be a context window of one billion tokens,
Why you may need a model that can handle context window with billions of tokens because you can use the model to handle thousands of prompts at the same time, but you will need a way for the model to share the knowledge without mixing the inputs and that if you want the model to act as a database or a search engine, or you’re dealing with large files like videos,
  
## Model identity

We will not use model fine-tuning to convert the model to a chatbot instead we will use prompt fine-tuning where the model weights will be frozen and the training goal is to get a learned system prompt or prefix values that makes this model output this response to this inquiry,  
And this will allow us to continue the training process on general content as much as we want, and will just use the learned system prompt before the user prompt,  

## Sparsity

You can decide the density or sparsity of the model at inference time like how many neurons you want to be active or used to work in the model out of the total number of neurons in the model,  
You can scale up your model after training by reducing sparsity and adding more dimensions or scale it down after training by increasing the sparsity and removing some dimensions,  
The brain is a fixed size neural network its size doesn't change based on the inputs size, but the amount of resources dedicated by the brain to a specific problem change based on the difficulty of the problem and that's possible because of the brain sparsity both in space and time,
  
## Runtime

Because we are using fixed network size instead of dynamical size, we can feed the inputs whole sequence in one shot to the model, run the model and get the outputs whole sequence in one shot, or we can feed the inputs word by word to the model and get the outputs word by word like how the transformers work,  
Also we can implement reasoning and thinking by giving the model various time steps to run by refeeding the model with its own outputs multiple times with backpropagation through time, depending on the length of the inputs and the outputs or the difficulty of the problem, so it can internally reason about the inputs before giving a final output, and the model doesn't need to output its chain of thoughts unless it was explicitly was told to do so,  

## Recursion

</br>

## Read & Write

</br>

## Partial training

</br>

## Blind training

</br>

## Self distillation

</br>

## General intelligence score with human evaluation

</br>

## General intelligence score with auto evaluation

</br>

## Implementation

I'll create two models using two different training techniques:  
Gradient Descent model and it will be mostly open source and for production purposes, and its implementation will be in  
<https://github.com/mohamed-services/mnn/tree/main/model/sgd>  
Experimental model and it will be mostly closed source and for experimentation purposes.  

Multidimensional layer implementation  
<https://github.com/mohamed-services/mnn/blob/main/layer.py>  
you are not bounded by this implementation, you can implement the multidimensional layer however you want based on your requirements, and decide what parameters to be shared and what to stay separate,
you can use multiple multidimensional layers in parallel like multi heads
you can stack multiple multidimensional layers for a deeper network
you can feed the outputs of the multidimensional layer to itself multiple times before passing the outputs to the next layer
this implementation is in TensorFlow, but you can convert it to PyTorch very easily
In this implementation every node is connected to itself multiple times which is inefficient in the parallel mode
I haven’t tested this code enough so it might be buggy
  
## Additional contributions from

<https://github.com/tagrib>  
<https://github.com/mgostIH>  
<https://github.com/bobbyiscool123>  

## You can join our Discord Server

<https://discord.com/channels/1366902833999511602/1366902833999511606>  

## In the end

This work is currently under development; I am focusing on resolving architectural challenges first before proceeding to the training stage. If you reached this part of the paper, I hope it was somewhat helpful to you.  
If you have any questions or recommendations on how to improve this architecture and willing to share it or make it open source, please contact me or send a pull request,  
If you have any freelancing task or any task that needs to be outsourced please contact me,  
You can email me on my personal email <mohamed.sourcing@gmail.com>
  
## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. Attention Is All You Need <https://arxiv.org/abs/1706.03762>  
[2] Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy. MLP-Mixer: An all-MLP Architecture for Vision <https://arxiv.org/abs/2105.01601>  
[3] Shikai Qiu, Andres Potapczynski, Marc Finzi, Micah Goldblum, Andrew Gordon Wilson. Compute Better Spent: Replacing Dense Layers with Structured Matrices <https://arxiv.org/abs/2406.06248>  
[4] Andres Potapczynski, Shikai Qiu, Marc Finzi, Christopher Ferri, Zixi Chen, Micah Goldblum, Bayan Bruss, Christopher De Sa, Andrew Gordon Wilson. Searching for Efficient Linear Layers over a Continuous Space of Structured Matrices <https://arxiv.org/abs/2410.02117>  

\
\
\
\
\
\
\
\

---

import string
from string import ascii_letters
import types
from typing import Iterable
import sys

def axis_call(x, w, axis, einsum):
    equation_x = ascii_letters[:len(x.shape)]
    if len(w.shape) == 2: #shared
        equation_w = equation_x[axis] 
    else: #separate
        equation_w = equation_x[1-len(w.shape):] 
    equation_w += ascii_letters[len(equation_x)]
    equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
    equation = equation_x + ',' + equation_w + '->' + equation_o
    #print(equation)
    return einsum(equation, x, w)

def layer_call(self, x):
    if self.execution_order == 'parallel':
        sub_layer = []
        for axis in self.w:
            sub_layer.append(axis_call(x, self.w[axis], axis, self.einsum))
        x = sum(sub_layer)
    else: # execution_order == 'sequential' or execution_order == 'single'
        for axis in self.w:
            x = axis_call(x, self.w[axis], axis, self.einsum)
    return x

def get_base_layer(backend, w_shapes, weights, **kwargs):
    """Dynamically imports and returns backend-specific base layer and einsum function."""
    backend_aliases = {
    'tensorflow': ['tensorflow', 'keras', 'tf'],
    'pytorch': ['torch', 'pytorch'],
    'jax': ['jax', 'flax'],
    }
    backend = backend.lower()
    if backend in backend_aliases['tensorflow']:
        try:
            import tensorflow as tf
            import keras
        except ImportError:
            raise ImportError("TensorFlow/Keras backend selected, but not installed.")
        layer = keras.layers.Layer(**kwargs)
        layer.call = types.MethodType(layer_call, layer)
        layer.einsum = tf.einsum
        layer.backend = 'tensorflow'
        layer.w = {}
        for axis in w_shapes:
            w_shape = w_shapes[axis]
            if weights:
                kernel_initializer = keras.initializers.Constant(weights[axis])
            else:
                kernel_initializer = keras.initializers.RandomUniform(minval=-sum(w_shape[:-1])**-0.5, maxval=sum(w_shape[:-1])**-0.5)
            layer.w[axis] = layer.add_weight(shape=w_shape, initializer=kernel_initializer, trainable=True, name=f'weight_{axis}')
        return layer
    elif backend in backend_aliases['pytorch']:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch backend selected, but not installed.")
        layer = torch.nn.Module(**kwargs)
        layer.forward = types.MethodType(layer_call, layer)
        layer.einsum = torch.einsum
        layer.backend = 'pytorch'
        layer.w = {}
        for axis in w_shapes:
            w_shape = w_shapes[axis]
            if weights:
                param = torch.nn.Parameter(torch.from_numpy(weights[axis]))
            else:
                param = torch.nn.Parameter(torch.empty(w_shape))
                torch.nn.init.uniform(param, a=-sum(w_shape[:-1])**-0.5, b=sum(w_shape[:-1])**-0.5)
            layer.register_parameter(f'weight_{axis}', param)
            layer.w[axis] = param
        return layer
    elif backend in backend_aliases['jax']:
        try:
            import jax
            import flax.linen as nn
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("JAX/Flax backend selected, but not installed.")
        layer = nn.Module(**kwargs)
        layer.__call__ = types.MethodType(layer_call, layer)
        layer.einsum = jax.numpy.einsum
        layer.backend = 'jax'
        layer.w = {}
        for axis in w_shapes:
            w_shape = w_shapes[axis]
            if weights:
                init_fn = nn.initializers.constant(weights[axis])
            else:
                init_fn = nn.initializers.uniform(sum(w_shape[:-1])**-0.5) 
            layer.w[axis] = layer.param(f'weight_{axis}', init_fn, w_shape)
        return layer
    else:
        raise ValueError(f"Unsupported backend: '{backend}'. Supported backends are: {', '.join(list(backend_aliases.keys()))}.")

def get_mode(shape, mode, execution_order, sequential_order, single_axis, axis_output):
    if type(mode) == str:
        mode = [mode.lower() for _ in shape]
    elif len(list(mode)) == 1:
        mode = [mode[0].lower() for _ in shape]
    elif len(list(mode)) != len(shape):
        raise ValueError(f"Length of mode list ({len(mode)}) must match length of shape ({len(shape)}).")
    if sequential_order.lower() == 'ascending':
        axes = range(-len(shape),0) 
    elif sequential_order.lower() == 'descending':
        axes = reversed(range(-len(shape),0))
    else:
        raise ValueError('sequential_order value must be ascending or descending')
    # validate execution input value
    execution_order = execution_order.lower()
    if execution_order not in ['parallel', 'sequential', 'single']:
        raise ValueError('execution_order value must be parallel or sequential or single')
    if execution_order == 'single':
        if single_axis is None or axis_output is None or single_axis >= len(shape) or single_axis < -len(shape):
            raise ValueError('single_axis and axis_output must be provided and valid')
        else:
            axes = [min(single_axis, single_axis-len(shape))]
    return axes, mode, execution_order

def __init__(shape: list|tuple|Iterable, # shape of the input, must be a list of integers, doesn't include the batch size
        backend: str, 
        mode :str|list[str]='separate', 
        execution_order :str='parallel', 
        sequential_order: str='ascending', 
        single_axis :int|None=None, 
        axis_output :int|None=None, 
        weights :dict|None=None, 
        **kwargs):
    # validate shape input value is a list 
    try:
        shape = list(shape) 
    except Exception as e:
        raise ValueError(f"shape must be a list or tuple or iterable, got {type(shape)}") from e
    # validate mode input value
    axes, mode, execution_order = get_mode(shape, mode, execution_order, sequential_order, single_axis, axis_output)
    # calculate the weights shapes for each axis
    w_shapes = {}
    for axis in axes:
        if mode[axis] == 'shared':
            in_shape = [shape[axis]]
        elif mode[axis] == 'separate':
            in_shape = list(shape)
        else:
            raise ValueError('mode value invalid, valid mode values are shared or separate or mixture list of both')
        out_shape = [shape[axis]]
        if execution_order == 'single':
            out_shape = [axis_output]
        w_shapes[axis] = in_shape+out_shape
    # validate backend input value and create the layer instance 
    layer = get_base_layer(backend, w_shapes, weights, **kwargs)
    # set execution order 
    layer.execution_order = execution_order
    return layer





# tensorflow implementation
# import tensorflow and keras if they are available
# you must import the backend before importing this module
if 'tensorflow' in sys.modules and 'keras' in sys.modules:
    import tensorflow as tf
    import keras

class MNN_tf(keras.layers.Layer):
    def __init__(self, shape, mode :str|list[str]='separate', execution :str='parallel', sequential_order: str='ascending', single_axis :int|None=None, axis_output :int|None=None, kernel_initializer=None, kernel_regularizer=None, kernel_constraint=None, weights=None, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape # shape of the input, must be a list of integers, doesn't include the batch size
        self.execution = execution.lower()
        if self.execution not in ['parallel', 'sequential', 'single']:
            raise ValueError('execution value must be parallel or sequential or single')
        if weights:
            self.w = weights
            return
        if self.execution == 'single':
            if single_axis is None or axis_output is None or single_axis >= len(shape) or single_axis < -len(shape):
                raise ValueError('single_axis and axis_output must be provided and valid')
        if type(mode) == str:
            mode = [mode.lower() for i in shape]
        elif len(mode) == 1:
            mode = [mode[0].lower() for i in shape]
        if len(mode) != len(shape):
            raise ValueError(f"Length of mode list ({len(mode)}) must match length of shape ({len(self.shape)}).")
        if sequential_order.lower() == 'ascending':
            axes = range(-len(self.shape),0) 
        elif sequential_order.lower() == 'descending':
            axes = reversed(range(-len(self.shape),0))
        else:
            raise ValueError('sequential_order value must be ascending or descending')
        if self.execution == 'single':
            axes = [min(single_axis, single_axis-len(shape))]
        self.w = {}
        for axis in axes:
            if mode[axis] == 'shared':
                in_shape = [shape[axis]]
            elif mode[axis] == 'separate':
                in_shape = list(shape)
            else:
                raise ValueError('mode value invalid, valid mode values are shared or separate or mixture list of both')
            if self.execution == 'single':
                self.w[axis] = self.add_weight(shape=in_shape+[axis_output], initializer=kernel_initializer, regularizer=kernel_regularizer, constraint=kernel_constraint)
                break
            self.w[axis] = self.add_weight(shape=in_shape+[shape[axis]], initializer=kernel_initializer, regularizer=kernel_regularizer, constraint=kernel_constraint)
    def axis_call(self, x, w, axis):
        equation_x = string.ascii_letters[:len(x.shape)]
        if len(w.shape) == 2: #shared
            equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
        else: #separate
            equation_w = equation_x[1-len(w.shape):] + string.ascii_letters[len(equation_x)]
        equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
        equation = equation_x + ',' + equation_w + '->' + equation_o
        #print(equation)
        return tf.einsum(equation, x, w)
    def call(self, x):
        if self.execution == 'parallel':
            sub_layer = []
            for axis in self.w:
                sub_layer.append(self.axis_call(x, self.w[axis], axis))
            x = sum(sub_layer)
        elif self.execution == 'sequential' or self.execution == 'single':
            for axis in self.w:
                x = self.axis_call(x, self.w[axis], axis)
        else:
            raise ValueError('execution value must be parallel or sequential or single')
        return x


# torch implementation
# import torch if it is available
# you must import the backend before importing this module
if 'torch' in sys.modules:
    import torch

class MNN_torch(torch.nn.Module):
    def __init__(self, shape, view, execution='parallel', sequential_order: str='ascending', **kwargs):
        super().__init__()
        self.shape = shape
        self.execution = execution.lower()
        if self.execution not in ['parallel', 'sequential']:
            raise ValueError('execution value must be parallel or sequential')
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
            param = torch.nn.Parameter(torch.empty(in_shape+[shape[axis]]))
            torch.nn.init.xavier_uniform_(param)
            param_name = f'weight_{axis}'
            self.register_parameter(param_name, param)
            self.w[axis] = param
    def axis_call(self, x, w, axis):
        equation_x = string.ascii_letters[:len(x.shape)]
        if len(w.shape) == 2: #shared
            equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
        else: #separate
            equation_w = equation_x[1-len(w.shape):] + string.ascii_letters[len(equation_x)]
        equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
        equation = equation_x + ',' + equation_w + '->' + equation_o
        #print(equation)
        return torch.einsum(equation, x, w)
    def forward(self, x):
        if self.execution == 'parallel':
            sub_layer = []
            for axis in self.w:
                sub_layer.append(self.axis_call(x, self.w[axis], axis))
            x = sum(sub_layer)
        elif self.execution == 'sequential':
            order = self.w if self.sequential_order == 'ascending' else reversed(self.w)
            for axis in order:
                x = self.axis_call(x, self.w[axis], axis)
        return x


# jax implementation
# import jax and flax if they are available
# you must import the backend before importing this module
if 'jax' in sys.modules and 'flax' in sys.modules:
    import jax.numpy as jnp
    from flax import linen as nn
    from typing import Union, List

class MNN_jax(nn.Module):
    shape: tuple
    view: Union[str, List[str]]
    execution: str = 'parallel'
    sequential_order: str = 'ascending'
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    def setup(self):
        # Process view configuration
        if isinstance(self.view, str):
            processed_view = [self.view.lower() for _ in self.shape]
        elif len(self.view) == 1:
            processed_view = [self.view[0].lower() for _ in self.shape]
        else:
            processed_view = [v.lower() for v in self.view]
        # Validate configuration
        if self.execution not in ['parallel', 'sequential']:
            raise ValueError("Execution must be 'parallel' or 'sequential'")
        if self.sequential_order not in ['ascending', 'descending']:
            raise ValueError("Sequential order must be 'ascending' or 'descending'")
        # Create parameters as attributes
        self.axes = list(range(-len(self.shape), 0))
        for axis in self.axes:
            view_type = processed_view[axis]
            if view_type == 'shared':
                in_shape = (self.shape[axis],)
            elif view_type == 'separate':
                in_shape = self.shape
            else:
                raise ValueError(f"Invalid view: {view_type}. Use 'shared' or 'separate'")
            param_shape = in_shape + (self.shape[axis],)
            setattr(self, f'w_{axis}', self.param(f'w_{axis}', self.kernel_init, param_shape))
    def axis_call(self, x, w, axis):
        equation_x = string.ascii_letters[:x.ndim]
        if len(w.shape) == 2:  # shared weights
            equation_w = equation_x[axis] + string.ascii_letters[len(equation_x)]
        else:  # separate weights
            # Calculate starting index for equation_w
            start_idx = x.ndim - len(w.shape) + 1
            equation_w = equation_x[start_idx:] + string.ascii_letters[len(equation_x)]
        equation_o = equation_x.replace(equation_x[axis], equation_w[-1])
        equation = f"{equation_x},{equation_w}->{equation_o}"
        return jnp.einsum(equation, x, w)
    def __call__(self, x):
        # Collect parameters in order
        params = [getattr(self, f'w_{axis}') for axis in self.axes]
        if self.execution == 'parallel':
            outputs = [self.axis_call(x, w, axis) for w, axis in zip(params, self.axes)]
            return sum(outputs)
        elif self.execution == 'sequential':
            order = zip(params, self.axes)
            if self.sequential_order == 'descending':
                order = reversed(list(order))
            for w, axis in order:
                x = self.axis_call(x, w, axis)
            return x

---
