# Modular Neural Networks

Mohamed Ibrahim

## Abstract

This is an open source paper so anyone is more than welcome to clone it, improve upon it and send a pull request to modify it.\

## Introduction

The purpose of this paper is not to prove point/s nor to introduce some new discovery, the purpose of this paper is to help create a full general purpose AI model on your personal computer from scratch and scale it based on your requirements whatever it is and get similar results to some of the Frontier models.\
It needs to be able to run continuously for many hours, days, weeks, and months, which is not possible using current architucures like trasnformer.\
It shouldn't take a whole data center to train an LLM, it should be possible to train an LLM on a single GPU.\
So to start we need first a good architecture.\
Also be cautious about any mathematical calculation, code snippet or any unproved claim in this paper, as maybe further verification is needed.\

## Modularity

Modularity means the ability to add or remove heads or experts withouting destroying the model.\
every expert must get a trainable 3 linear layer as an attachment to convert data from the shared space representation to the expert representation, and to convert the data from the expert representation to the shared space representation.\
outputs = linear_2(expert(linear_1(inputs))) + linear_3(inputs)\
This architecture will allow the usage experts or heads from different open-weights models from different sources without needing to retrain any of those experts by freezing the expert weights and just training the linear layers for that expert to be able to communicate with the shared space.\

## Multidimensionality

If we are using one dimensional neural network like MLP architecture for our model then, if we have an image of size (600, 600, 3) then its flatten context window is approximately 1 million token then we need 1 trillion parameters for a single layer to process this image, also if we are dealing with a text of size 16k words and embedded every word in a vector of size 64 so its flatten context window will be approximately 1 million token then\we need 1 trillion parameters per layer to process this text, and the truth is most of those 1 trillion parameters doesn’t contribute much and we can convert a lot of those parameters to zeros without affecting the final results much, So the problem of MLP is that it scales quadratically that’s why we are not using\MLP for tasks of large size like images or text,\
The transformer models decrease this problem by reshaping the flat inputs like text into 2d arrays for example, from 1 million to (16384, 64) for that text input, and cut and reshape the image into patches for example from 1 million to (5625, 192), so the computational complexity is reduced from 1 million squared to\(32768^2, 32^2) and (5625^2, 192^2),\The logical conclusion is that we should not stop at just two dimensions as increasing the dimensions reduces the needed computation, so we can reduce the computation by reshaping the inputs from two dimensions to three or more, if we want we can reshape the inputs to a maximum of dimensions equals\log_2 of its size, so every dimension will be of size two,\
The transformer model can process the inputs in multidimensional space, for example if you have input sample of ten thousand words and embedding of size 64 usually you can run the attention mechanism on the first axis of size 10,000 and then you will run the feedforward layer on the second axis of 64, or you can reshape this sample to be (100, 100, 64) and run the attention mechanism on the first axis of size 100 then run the attention mechanism on the second axis of size 100 and then run the feedforward layer on the third axis of size 64, another example if you have an input of size one million words then you can run the attention mechanism on the first axis of size 1,000,000 and run the feedforward layer on the second axis of size 64, or you can reshape it to (100, 100, 100, 64) and then run the attention mechanism of size 100 on the first axis then run it on the second axis of size 100 then run it on the third axis of size 100 then run the feedforward layer on the fourth axis of size 64, I’ll not use the attention mechanism in my model, I’m mentioning it just in case you want to use it,\
You can implement multidimensional layers using attention layers or convolutional layers, or locally connected layers or feedforward layers or many other options.\

## True meaning of multidimensionality

For attention mechanism from 1d to 2d\
1d means 1 million squeared in one step equals 1 trillion oprations\
2d means 2 one thousand squared in two steps equals 4 million operations\
250 X reduction by going one step further\

## Feedforward neural networks

Feedforward layers, is a static or fixed size type of layers where you must decide the input size and the output size of the layer before creating it, which is different from the dynamic range that the attention mechanism gives us, theoretically you can give the transformer model an insanely large input in size and it will compute it and return you the output, but realistically most LLM platforms and transformer models out there have limits to the input size that you can feed to the model and the output size that you can get from the model, right now most of the models have limited context window that’s less than one million words, and some below hundred thousand words,\and as soon as you decide the maximum size for your model whatever it is, then the attention mechanism isn't needed anymore and you can just use feedforward layer instead of it and zero pad your inputs to the needed size, so instead of using the attention mechanism on the first axis and the feedforward layer on the second axis, you can just use feedforward layer on the first axis and another feedforward layer on the second axis and you will get the same outputs, also you can make the feedforward layer sparse or more dynamic by deciding how many nodes you want to be used for any specific input out of the total number of nodes in that layer based on input size or the task difficulty,\

## Multidimensional layers

Multidimensional neural network, is like any ordinary feedforward neural network,\
If we have one fully connected layer of one hundred neurons with input size of one hundred then this fully connected layer will have ten thousand connections (100^2), We can reduce the number of parameters while keeping this layer fully connected by reshaping the input from one dimension of one hundred to two dimensions of ten by ten and run ten different dense layers with ten neurons inside on each row and run another ten dense layers with ten neurons inside on each column and sum the two sets, then the number of the parameters will be (10^3)+(10^3) which is equal to 2,000 parameters but because we converted the 1d input to 2d then we need to run the sublayers on the two axes sequentially over two steps or we will need to stack two layers, because the maximum number of steps needed to move information from any point to any point using a straight line in a multidimensional array is equal to the number of dimensions so we need one step for 1d and two steps for 2d and three steps for 3d and so on,\
And those numbers are for non-shared or spatially separate parameters, but if we will share the parameters inside every axis the number of parameters will be ((10^2) + (10^2)) = 200 shared parameters, also if all the axes are of the same size then we can share the parameters of one axis with all the axes and the number of parameters will be (10^2) = 100, and in general you can decide what parameters you want to be shared inside your model and how it will be shared between the neurons and what parameters you don’t want to be shared inside your model based on your requirements,\

## Layer resizing

The multidimensional layers give you the freedom to resize the layers shape whatever you want by using a dense layer on the specific axis you want to resize, like how the feedforward layers allows you to get different output size than the input,\

## Biases

Note that the biases are not accounted for in the above calculations because if you will use an embedding layer as an input layer then using the biases is optional inside the hidden layers, but you’re free to use biases in your model,\

## Mixture of experts

The attention mechanism itself isn't needed because you can use a mixture of experts instead of it to route the data between the tokens also you can choose the size of the 2d matrix from n by 2 to n by (n-1) then you can just sum all the data along the second dimension, Also you can use the mixture of experts on a higher dimensions.\
Also you can use the attention mechanism on higher dimensions.\
You can also use multi-agents or mult-heads where every agent or every head process a part of the input or a part of the output and they are communicating using addressess so every agent can decide which agents it wants to send data to and which agents it wants to ask data from.\
So agents works like workers in a factory where every worker do his part.\
And they are rewarded for distributing work and loads equaly and for being cooperative.\
The addressing system:\
1D (all agents will be on a line or a circle).\
Modular (address 13 for network of size 5 will be 13 mod 5 so thd real address will be 3).\
Binary (addresses will be an n bit binary positions for example 32 bit so the agents can address each other efficiently and accurately).\
Signed (address can positive 9 or negative 9, like 9 move to the right or 9 moves to the left relative to the current position).\
Relative or absolute (we can use relative or absolute positioning).\
Agents talk every few time steps.\
Mixture of experts architecture can be used or should be used.\
Some agents can act as an input layers, some agents can act as a hidden layers, and some agents can act as an output layers.\
Experts can have different numbers of dimensions for example two, three, or four as soon as tolens size is smaller than input/output maximum size so the expert can reshape the data to its number of dimensions.\
The first n bits will be used to address other agents for example the first 1024 bits will be used to address another 32 agents.\
A subset of the agents can be grouped in a small pool to work on a smaller task.\
True mixture of experts architecture.\
Federated machine learning where any one or any company can train their private agent and integrate it with publicly available agents or experts.\
True Mixture of experts architecture\
No attention needed As it turns out that attention is not necessary in large language models.\
Can we use a central memory for coordinating something like RAG where every agent outputs a small vector\
And after that nearest neighbors algorithm will be used to to determine the set of inputs that will be used for the next time step.\
Make every expert output two vectors the first to be its axon central location and the other to be its dendrites central location and this cordinates can be 3d or 4d or 8d cordinates.\
Also agents can choose the radius that they want the nearest neighbors to be viewed.\
Make the current weights of similar size like 8,8,8,8 and group them in one numpy file to be a single expert.\
The system can be asynchronous so every agent can read or write on his own speed for example an agent can run for five steps then read or write and another agen can for eight steps then read or write.\
RAG memories are inherently builtin in the system so a memory can live as an input or output without changing.\
The agent can output floating point based cordinates or text based cordinates.\
agents can be mnn or conv or rnn or transformers.\
You can convert convert the distance between the cell and its neighbours into weights where's a lower distance means a higher weight.\
w = 1 - current_distance / sum(all distances)\
Any head can act as input head, hidden head, or output head.\

## Nearest Neighbors

Using the weighted k nearest neighbors algorithm \
By calculating the distance between the heads coordinates to the current head coordinates and normalizing the values between zero and one and subtracting the normalized values from one then multiplying every input by its value or weight. By doing that we teach the networks if it wants to decrease the weight for an input from a head then it should increase the distance or the difference between the current head coordinates and that head coordinates and if it wants to increase the wight for that input then it should decrease the distance or the difference between the coordinates.\
The k nearest neighbors algorithm is imbalanced which means that not all the heads or tokens will be represented equally or the same number of times which means that some tokens can be overrepresented or underrepresented or not represented at all. To fix this issue you can enforce that any token must be represented m number of times to its nearest neighbors even if it isn't in the top k nearest neighbors.\
Add nearest neighbors attention to the paper.
The head will generate the query and the key an the value.\
The size of the queries and keys is four or eight values.\
The model will use the k-nearest neighbors approximation to find the nearest keys to the current query.\
Distance(abs(query - keys))\
Normalize the distance \
Invert the normalized distance \
Multiply the inverted normalized distances with the selected values.\
Standard Attention: "I am the query. I will look at all keys, calculate a similarity score for each, and take a weighted average of all values."\
Nearest Neighbor Attention: "I am the query. I will find the top-$k$ keys that are closest to me in vector space, and I will only attend to those."\

## Activation function

The following activation functions are the candidates for the mnn model\
relu: pros (simple to compute, sparse, recommended overall) cons (sharp, dying nodes)\
leakyrelu: leakyrelu(x) = max(x, 0.25*x)\
hardtanh: hardtanh(x) = clip(x, -1, 1). recomended if you will quantize your model and will use fixed point arithmetic as it more compatible with a Fixed-point arithmetic by making the minimum and the maximum limit of the datatype the same as the activation function for example an 8 bit datatype will have 256 values from approximatly negative one to approximatly positive one and any lower or higher value will be clipped naturally by the datatype\
elu: elu(x) = x if x >= 0 else exp(x) - 1\
melu: melu(x) = x \* exp( min(x,0)). melu is a variant of activation functions like gelu, silu, and mish but it is designed for deeper networks as it have better gradients flow compared to gelu, silu, and mish\
sort2: sort2(x) = reshape(x, [-1, 2]); sort(x, axis=-1); reshape(x, original_shape)\
modified norm: x \* abs(x / sqrt( sum( square(x), axis=-1, keepdims=True)))
Some models may perform better with a partial activation instead of full activation, I think you should give it a try, for example you can apply the activation function on the first three quarters of the nodes and leave the remaining quarter as linear as it is, because the network needs linear and non-linear information to be passed from layer to layer, so by keeping some outputs linear you improve the forward and backward data flow in your model,\

## Selective relu gradients

Solving the dying relu problem\
If the relu output > 0:\
   Apply the gradients to the layer weights\
If the relu output == 0:\
  If gradient is Negative: Apply a small fraction of the gradients to the layer weights (Allow the neuron to climb back up to 0).\
  If gradient is Positive: Block it (Prevent the neuron from being pushed further down).\
Return the normal unmodified relu gradients to the preceding layers or steps to avoid unstablizing them by the fake gradients.\

## Backpropagation through time

We can use backpropagation through time without storing all the intermediate values in memory by using leaky relu and making sure that the weights matrix is Invertible\
Recommended values for leaky relu alpha are (0.125 , 0.25 , 0.5)\
In the backwards pass\
We reverse the leaky relu by multiplying the negative outputs with (1/alpha) to get the activation function inputs.\
Then we matrix multiply the linear layer outputs with the inverse of the weights to get the layer inputs.\
And so on step by step back in time.\
You need to store the intermediate values every 16 or 32 time steps to accommodate for the accumulated floating point errors.\\
You can do the same with any monotonic activation function like elu.\
You can also do the same with the abs activation function but you need to store the sign of the values in every intermediate step.\
Also you can do the same with the sort activation function but you need to store the original order of the values.\
You can backprop sort2 activation through time by storing the order of every two outputs using one bit and bundle every eight bits in one byte, so you compress the size of the intermediate values by one over sixty four.\
You can do the same with abs activation function but the compression rate would be one over thirty two.\

## Connectivity

There're a lot of ways to wire a multidimensional layer for example you can make it fully wired or wire it randomly or wire nodes to their close neighbors using sliding n-dimensional convolution kernels, or using neural architecture search to find the best wiring pattern, you can use any dense or sparse wiring technique that suits your targets, I’ll will wire my model using axis based point of view like using rows and columns etc., so every node will be connected to every other node that shares all the dimensional coordinates or indexes with it but doesn’t share exactly one coordinate or index with it as explained above,\

## Network design

you're free to design the network the way you want by choosing how many dimensions and the size of every dimension also the number of the hidden layers and what kind of layers to include and what not to include so you're not bounded by any design pattern, and for my model I’ll use one multidimensional layer with separate or non-shared parameters , I’ll start with a small context window and will scale the model up slowly while training, also the previous outputs will be refed to the network multiple times before getting the final outputs, I know that I should use shared parameters architecture with lower number of dimensions and large dimension size and multiple hidden layers and that is the right way to do it, but I'll use fully separate parameters architecture for my model because it will be more suitable for the training techniques that I’ll use later,\

## Tokenization

I'll use byte based tokenization for all text modality.\
We will have three types of data:\
Binary data.\
Discrete data like text and will Use embedding of size 16 for every 256 discrete values.\
Signal or continuous data like pixel colors intensities and audio waves amplitudes.\
Text data like characters can processed via embedding and reverse embedding layer.\
As soon as the embedding size is 256 or below then you're winning performance against the word embedding style.\
Image data like color channels can be represented via continuous floating point values between negative one and one. And the three colors channels can be represented using three values or more.\
You can have multiple embedding layers for the same modality and you can choose between them like mixture of experts.\
The input embedding and the network outputs must be the same, in other words you must use the same layer weights for embedding and unembedding.\
Every token will be preceded by an 8 bit section for Positional Encoding and for determining the modality like text or image or sound or etc.\

## Padding

Transformer models gives flexible context window, but the feedforward models needs fixed context window so we have to pad our inputs to a fixed size so the feedforward layers can process it, you can leave the beginning of your inputs and only pad the end of the inputs, but for me all the inputs will be padded with random number of zeros before it and the inputs will be zero padded after it till the end of the model input size so the position of the input will be random not in the beginning of the context window,\

## Context window

I’ll start my model with a context window of one character or one pixel then I’ll start to scale it up and increase the context window,\
My final goal will be a context window of one billion tokens,\
Why you may need a model that can handle context window with billions of tokens because you can use the model to handle thousands of prompts at the same time, but you will need a way for the model to share the knowledge without mixing the inputs and that if you want the model to act as a database or a search engine, or you’re dealing with large files like videos,\

## Model identity

We will not use model fine-tuning to convert the model to a chatbot instead we will use prompt fine-tuning where the model weights will be frozen and the training goal is to get a learned system prompt or prefix values that makes this model output this response to this inquiry, so the model name itself will be learned,\
And this will allow us to continue the training process on general content as much as we want, and will just use the learned system prompt before the user prompt,\

## Sparsity

You can decide the density or sparsity of the model at inference time like how many neurons you want to be active or used to work in the model out of the total number of neurons in the model,\
You can scale up your model after training by reducing sparsity and adding more dimensions or scale it down after training by increasing the sparsity and removing some dimensions,\
The brain is a fixed size neural network its size doesn't change based on the inputs size, but the amount of resources dedicated by the brain to a specific problem change based on the difficulty of the problem and that's possible because of the brain sparsity both in space and time,\

## Runtime

Because we are using fixed network size instead of dynamical size, we can feed the inputs whole sequence in one shot to the model, run the model and get the outputs whole sequence in one shot, or we can feed the inputs word by word to the model and get the outputs word by word like how the transformers work,\
Also we can implement reasoning and thinking by giving the model various time steps to run by refeeding the model with its own outputs multiple times with backpropagation through time, depending on the length of the inputs and the outputs or the difficulty of the problem, so it can internally reason about the inputs before giving a final output, and the model doesn't need to output its chain of thoughts unless it was explicitly was told to do so,\

## Runtime learning

You can have run time learning using fast weights with Hebbian Updates\
Designing a layer that utilizes "Fast Weights" to store short-term context. The network should have a fixed weight matrix (Wslow​) and a plastic weight matrix (Afast​) that is updated on the fly.\
The Afast​ matrix updates based on the synchrony between inputs and the ReLU output.\
Strengthen Rules: If inputs and outputs are in sync (both active or both inactive), increase the associative weight strength.\
Weaken Rules: If inputs and outputs are out of sync (one active, one inactive), decrease the weight.\
Ensuring that Afast​ decays over time to prevent runaway values, and combines with Wslow​ for the final forward pass.\

## Recursion

The model is recursive neural network not recurrent neural network.\

## Read & Write

The model can read the inputs in multiple time steps like batchs or can read all the inputs at once, also the model can write the outputs as batchs or all at once.\
Also the model is interactive which means you can give it inputs and get outputs from it then give it more inputs.\

## Partial training

Training the model with limited backpropagation through time by only tracing the gradients back trhough time to a number of time steps smaller than the total number of the time steps.\

## Zeroth and first order training

Zeroth order is training the model without backpropagation using random gradients.\

## Self distillation

Self distillation is running the model for long steps to an outputs then train the model to produce those outputs with lower number of steps.\

## Training methods

An autoencoder style training will be used.\
Which means no next word prediction instead the model will learn to compress the input and recover missing data or denoise the data that was noised from the other samples.\
And training methods will be.\
One input to one output.\
One input to many outputs.\
Many inputs to one output.\
Many inputs to many outputs.\

## Training data

No human labels will be used in the pretraining of the model, only raw unsupervised learning and reinforcement learning.\
Supervised learning will be only used in the prompt fine-tuning.\
Only open source data will be used in the model training\

## Video processing

Vedio proccessing using multidimensional attention and i frames and epsoides\

## Lossy weights compresion

Algorithms like DCT, quantization and run length encoding can be used to compress the model weights.\

## General intelligence score

The model should be tested againest data that it can't reach via interpolation or extrapolation.\

## Implementation

I'll create two models:\
Gradient Descent model and it will be mostly open source and for production purposes, and its implementation will be in\
[Model Directory](/model)\
Experimental model and it will be mostly closed source and for experimentation purposes.\\
Multidimensional layer implementation\
[Layer Implementation](/layer.py)\
you are not bounded by this implementation, you can implement the multidimensional layer however you want based on your requirements, and decide what parameters to be shared and what to stay separate,\
you can use multiple multidimensional layers in parallel like multi heads\
you can stack multiple multidimensional layers for a deeper network\
you can feed the outputs of the multidimensional layer to itself multiple times before passing the outputs to the next layer\
this implementation is in TensorFlow, but you can convert it to PyTorch very easily\
In this implementation every node is connected to itself multiple times which is inefficient in the parallel mode\
I haven’t tested this code enough so it might be buggy\

## You can join our Discord Server

[Discord](https://discord.com/channels/1366902833999511602/1366902833999511606)

## In the end

This work is currently under development; I am focusing on resolving architectural challenges first before proceeding to the training stage. If you reached this part of the paper, I hope it was somewhat helpful to you.\
If you have any questions or recommendations on how to improve this architecture and willing to share it or make it open source, please contact me or send a pull request,\
If you have any freelancing task or any task that needs to be outsourced please contact me,\
You can email me on my personal email <mohamed.sourcing@gmail.com>

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. Attention Is All You Need <https://arxiv.org/abs/1706.03762>\
[2] Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy. MLP-Mixer: An all-MLP Architecture for Vision <https://arxiv.org/abs/2105.01601>\
