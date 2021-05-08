---
header:
  overlay_color: "#333"
title: "Unwrapping the Hidden states of RNN models"
data: 2019-03-11
tags: [machine learning, data science, deep learning, neural network, recurrent neural network]
excerpt: "Deep Learning, Data Science, RNN"
mathjax: "true"
---

This post is about detailed view of hidden states, how they change in case of unidirectional rnn's, bidirectional rnn's, single layer, multi layers etc. 

Understanding how the hidden states and outputs of RNN and the relation between them is a bit confusing for the beginners to understand. We will go step by step, unwrap them and see how they are related, how they change.

I hope you know about RNN's. If not refer to this link: [Understanding LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

*The full code is available in my github repo:* [link](https://github.com/graviraja/pytorch-sample-codes/blob/master/hidden_rnn.py)

We will devide this post into 4 sections.
- Single layer, uni-directional RNN
- Multi layer, uni-directional RNN
- Single layer, bi-directional RNN
- Multi layer, bi-directional RNN

We will be using PyTorch for coding purposes.

Before going into each of the section, we will first the see the basic equations of RNN.

The input to the RNN is a Sequence $$X = \{x_1, x_2,...., x_t\}$$ and the hidden states, $$H = \{h_1, h_2,...., h_t\}$$ are calcualted using the following equation:

$$h_t = RNN(x_t, h_{t-1})$$

In general, the outputs $$O = \{o_1, o_2,.....,o_t \}$$ are calculated using the following equation:

$$o_t = ReLU(Linear(h_t))$$

**Note: There is a small change in implementation of RNN when using PyTorch. Output is not calculated through the linear and relu functions, it is the same $$h_t$$, i.e $$o_t = h_t$$**

## Initial Setup

Let's create some dummy data, which can be used for understanding the above mentioned sections.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import LongTensor

# create a dummy data of batch_size = 3
data = ['long_str', 'tiny', 'medium']

# create the vocabulary
vocab = ['<pad>'] + sorted(set([char for seq in data for char in seq]))
# vocab = ['<pad>', '_', 'd', 'e', 'g', 'i', 'l', 'm', 'n', 'o', 'r', 's', 't', 'u', 'y']

# convert into numerical form
vectorized_data = [[vocab.index(tok) for tok in seq] for seq in data]
# vectorized_data = [[6, 9, 8, 4, 1, 11, 12, 10], [12, 5, 8, 14], [7, 3, 2, 5, 13, 7]]

# prepare data, by padding with 0 (<pad> token), making the batch equal lengths
seq_lengths = LongTensor([len(seq) for seq in vectorized_data])
sequence_tensor = Variable(torch.zeros(len(vectorized_data), seq_lengths.max(), dtype=torch.long))

for idx, (seq, seq_len) in enumerate(zip(vectorized_data, seq_lengths)):
    sequence_tensor[idx, :seq_len] = LongTensor(seq)

# sequence_tensor = ([[ 6,  9,  8,  4,  1, 11, 12, 10],
#                     [12,  5,  8, 14,  0,  0,  0,  0],
#                     [ 7,  3,  2,  5, 13,  7,  0,  0]])

# convert the input into time major format
sequence_tensor = sequence_tensor.t()
# sequence_tensor shape => [max_len, batch_size]

input_dim = len(vocab)
print(f"Length of vocab : {input_dim}")

# hidden dimension in the RNN
hidden_dim = 5

# embedding dimension
embedding_dim = 5

```

![vocab figure](/assets/images/rnn_hidden/vocab.png "Vocab Figure")

## Single Layer, Uni-Directional RNN

![sign uni figure](/assets/images/rnn_hidden/single_uni.png "RNN Figure")

The input $$X = \{x_1, x_2, x_3, x_4\}$$ is passed through the RNN, and the outputs and hidden states are calcualted using the above equations.

When passed the RNN, it returns output for each time step i.e $$\{o_1^1, o_2^1, o_3^1, o_4^1\}$$ and the final hidden state i.e $$\{h_4^1\}$$

### Relation

*The relation between the outputs and hidden state returned by the RNN is the final output is same as the final hidden state. ($$o_4^1 == h_4^1$$)*

### Implementation

Considering the time-major format, shape of the inputs and outputs of RNN are as follows:

> Input shape: [max_len, batch_size]

>Output shape: [max_len, batch_size, hidden_size]

>Hidden shape: [1, batch_size, hidden_size]

Create the single layer, unidirectional RNN class.


```python
class Single_Layer_Uni_Directional_RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)

    def forward(self, input):
        # input shape => [max_len, batch_size]

        embed = self.embedding(input)
        # embed shape => [max_len, batch_size, embedding_dim]

        output, hidden = self.rnn(embed)
        # output shape => [max_len, batch_size, hidden_size]
        # hidden shape => [1, batch_size, hidden_size]

        return output, hidden
```

Now let's create the instance of the class and compare the output and hidden states.

```python

n_layers = 1
bidirectional = False
model = Single_Layer_Uni_Directional_RNN(input_dim, embedding_dim, hidden_dim, n_layers, bidirectional)
output, hidden = model(sequence_tensor)

print(f"Input shape is : {sequence_tensor.shape}")
print(f"Output shape is : {output.shape}")
print(f"Hidden shape is : {hidden.shape}")

assert (output[-1, :, :] == hidden[0]).all(), "Final output must be same as Hidden state in case of Single layer uni-directional RNN"

```
![shape figure](/assets/images/rnn_hidden/single_uni_shapes.png "Shape Figure")


## Multi Layer, Uni-Directional RNN

![multi uni figure](/assets/images/rnn_hidden/multi_uni.png "RNN Figure")

Let's consider a 2 layer RNN, and the concept is same for more layers.

The input $$X = \{x_1, x_2, x_3, x_4\}$$ is passed through the first layer of RNN, and the outputs of first layer are then passed as the inputs to the second layer RNN.

The outputs returned is the outputs of final layer of RNN. 

The hidden states are the final hidden state of each layer in RNN.

So the outputs are  $$\{o_1^2, o_2^2, o_3^2, o_4^2\}$$ and the final hidden state i.e $$\{h_4^1, h_4^2\}$$

### Relation

*The relation between the outputs and hidden state returned by the RNN is the final output is same as the final hidden state of the final layer. i.e ($$o_4^2 == h_4^2$$)*

### Implementation

Considering the time-major format, shape of the inputs and outputs of RNN are as follows:

> Input shape: [max_len, batch_size]

>Output shape: [max_len, batch_size, hidden_size]

>Hidden shape: [num_layers, batch_size, hidden_size]

Create the multi layer, unidirectional RNN class.

```python
class Multi_Layer_Uni_Directional_RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)

    def forward(self, input):
        # input shape => [max_len, batch_size]

        embed = self.embedding(input)
        # embed shape => [max_len, batch_size, embedding_dim]

        output, hidden = self.rnn(embed)
        # output shape => [max_len, batch_size, hidden_size]
        # hidden shape => [num_layers, batch_size, hidden_size]

        return output, hidden

n_layers = 2
bidirectional = False
model = Multi_Layer_Uni_Directional_RNN(input_dim, embedding_dim, hidden_dim, n_layers, bidirectional)
output, hidden = model(sequence_tensor)

print(f"Input shape is : {sequence_tensor.shape}")
print(f"Output shape is : {output.shape}")
print(f"Hidden shape is : {hidden.shape}")

assert (output[-1, :, :] == hidden[-1]).all(), "Final output must be same as Final Hidden state in case of Multi layer uni-directional RNN"

```
![shape figure](/assets/images/rnn_hidden/multi_uni_shapes.png "Shape Figure")

## Single Layer, Bi-Directional RNN

Now let's see the outputs and hidden states of Single Layer, Bi-Directional RNN.

![single bidir figure](/assets/images/rnn_hidden/single_bi.png "RNN Figure")

Same as before, the input to the RNN is $$X = \{x_1, x_2, x_3, x_4\}$$. The difference here is that, there are 2 RNN's. We call them Forward RNN (which reads input from left to right \{$$x_1 ... x_4$$\} ) and Backward RNN (which reads input from right to left \{$$x_4 ... x_1$$\}) 

Corresponding to 2 RNN's there will be 2 outputs (forward and backward) and 2 hidden outputs (forward and backward).

**Note: The naming convention of hidden states and outputs varies from place to place.**

- *We denote the hidden state $$\overrightarrow{h_t}$$ after reading the input $$x_t$$ from left to right*
- *We denote the hidden state $$\overleftarrow{h_t}$$ after reading the input $$x_t$$ from right to left*
- *We denote the output $$\overrightarrow{o_t}$$ after reading the input $$x_t$$ from left to right*
- *We denote the output $$\overleftarrow{o_t}$$ after reading the input $$x_t$$ from right to left*

The outputs returned by the RNN are stacked on top of each other (forward outputs and backward outputs).

$$o_t = [\overrightarrow{o_t}:\overleftarrow{o_t}]$$


The hidden states returned by the RNN are the final forward hidden state $$\overrightarrow{h_4^1}$$ and the final backward hidden state $$\overleftarrow{h_0^1}$$.

### Relation

Considering the time-major format, shape of the inputs and outputs of RNN are as follows:

> Input shape: [max_len, batch_size]

>Output shape: [max_len, batch_size, hidden_size * 2]

>Hidden shape: [num_dir, batch_size, hidden_size]

The Hidden shape is *[2, batch_size, hidden_size]*
- hidden[0] is Final Forward Hidden state $$\overrightarrow{h_4^1}$$
- hidden[1] is Final Backward Hidden state $$\overleftarrow{h_0^1}$$

Outputs are stacked on top of each other (forward and backward) for each time step.
The final time step's output contains the $$o_4^1 = [\overrightarrow{o_4^1}:\overleftarrow{o_4^1}]$$

First set of hidden_dim states in the final time step's output is same as the final forward hidden state, i.e *($$\overrightarrow{o_4^1} == \overrightarrow{h_4^1}$$)*

Last set of hidden_dim states in the initial time step's output is same as the final backward hidden state, i.e *($$\overleftarrow{o_1^1} == \overleftarrow{h_0^1}$$)*

### Implementation

Create the multi layer, unidirectional RNN class.

```python
class Single_Layer_Bi_Directional_RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)

    def forward(self, input):
        # input shape => [max_len, batch_size]

        embed = self.embedding(input)
        # embed shape => [max_len, batch_size, embedding_dim]

        output, hidden = self.rnn(embed)
        # output shape => [max_len, batch_size, hidden_size * 2] => since forward and backward outputs are stacked
        # hidden shape => [2, batch_size, hidden_size]

        return output, hidden

n_layers = 1
bidirectional = True
model = Single_Layer_Bi_Directional_RNN(input_dim, embedding_dim, hidden_dim, n_layers, bidirectional)
output, hidden = model(sequence_tensor)

print(f"Input shape is : {sequence_tensor.shape}")
print(f"Output shape is : {output.shape}")
print(f"Hidden shape is : {hidden.shape}")

assert (output[-1, :, :hidden_dim] == hidden[0]).all(), "First hidden_dim of output at last time step must be same as Final Forward Hidden state in case of Single layer bi-directional RNN"
assert (output[0, :, hidden_dim:] == hidden[-1]).all(), "Last hidden_dim of output at initial time step must be same as Final Backward Hidden state in case of Single layer bi-directional RNN"
```
![shape figure](/assets/images/rnn_hidden/single_bi_shapes.png "Shape Figure")

## Multi Layer, Bi-Directional RNN

Now let's see the outputs and hidden states of Multi Layer, Bi-Directional RNN.

![multi bidir figure](/assets/images/rnn_hidden/multi_bi.png "RNN Figure")

We will learn about 2 layer bi-directional RNN. The same concept can be applied to multiple layers.

The input to the RNN in layer 1 is $$X = \{x_1, x_2, x_3, x_4\}$$. The Forward RNN of layer 1 (reads input from left to right \{$$x_1 ... x_4$$\} ) and Backward RNN (reads input from right to left \{$$x_4 ... x_1$$\}).

The Forward and Backward RNN in the layer 1 outputs:
- Forward outputs:  $$\{\overrightarrow{o_1^1}, \overrightarrow{o_2^1}, \overrightarrow{o_3^1}, \overrightarrow{o_4^1}\}$$
- Backward outputs $$\{\overleftarrow{o_1^1}, \overleftarrow{o_2^1}, \overleftarrow{o_3^1}, \overleftarrow{o_4^1}\}$$


Which are inputs to the Forward RNN and Backward RNN of layer 2.

The outputs returned by the RNN are stacked on top of each other (forward outputs and backward outputs of layer 2). 

$$o_t = [\overrightarrow{o_t^2}:\overleftarrow{o_t^2}]$$


The hidden states returned by the RNN are the final forward hidden state $$\{\overrightarrow{h_4^1}, \overrightarrow{h_4^2}\}$$ and the final backward hidden state $$\{\overleftarrow{h_0^1}, \overleftarrow{h_0^2}\}$$ of each layer.

### Relation

Considering the time-major format, shape of the inputs and outputs of RNN are as follows:

> Input shape: [max_len, batch_size]

>Output shape: [max_len, batch_size, hidden_size * 2]

>Hidden shape: [num_layers * num_dir, batch_size, hidden_size]

The Hidden shape is *[2 * 2, batch_size, hidden_size]*

Let's view Hidden states as *[num_layers, num_dir, batch_size, hidden_size]*
- hidden[0][0] is Final Forward Hidden state of layer 1 : $$\overrightarrow{h_4^1}$$
- hidden[0][1] is Final Backward Hidden state of layer 1 : $$\overleftarrow{h_0^1}$$
- hidden[1][0] is Final Forward Hidden state of layer 2 : $$\overrightarrow{h_4^2}$$
- hidden[1][1] is Final Backward Hidden state of layer 2 : $$\overleftarrow{h_0^2}$$

Outputs are stacked on top of each other (forward and backward) for each time step of final layer.
The final time step's output contains the $$o_4^2 = [\overrightarrow{o_4^2}:\overleftarrow{o_4^2}]$$

First set of hidden_dim states in the final time step's output is same as the final forward hidden state of layer 2, i.e *($$\overrightarrow{o_4^2} == \overrightarrow{h_4^2}$$)*

Last set of hidden_dim states in the initial time step's output is same as the final backward hidden state of layer 2, i.e *($$\overleftarrow{o_1^2} == \overleftarrow{h_0^2}$$)*

### Implementation

Create the multi layer, bi-directional RNN class.

```python
class Multi_Layer_Bi_Directional_RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional)

    def forward(self, input):
        # input shape => [max_len, batch_size]

        embed = self.embedding(input)
        # embed shape => [max_len, batch_size, embedding_dim]

        output, hidden = self.rnn(embed)
        # output shape => [max_len, batch_size, hidden_size * 2] => since forward and backward outputs are stacked
        # hidden shape => [num_layers * 2, batch_size, hidden_size]

        return output, hidden

n_layers = 2
bidirectional = True
model = Multi_Layer_Bi_Directional_RNN(input_dim, embedding_dim, hidden_dim, n_layers, bidirectional)
output, hidden = model(sequence_tensor)

print(f"Input shape is : {sequence_tensor.shape}")
print(f"Output shape is : {output.shape}")
print(f"Hidden shape is : {hidden.shape}")

batch_size = sequence_tensor.shape[1]
hidden = hidden.view(n_layers, 2, batch_size, hidden_dim)
print(f"Reshaped hidden shape is : {hidden.shape}")

assert (output[-1, :, :hidden_dim] == hidden[n_layers - 1][0]).all(), "First hidden_dim of output at last time step must be same as Final Forward Hidden state of final layer in case of Multi layer bi-directional RNN"
assert (output[0, :, hidden_dim:] == hidden[n_layers - 1][1]).all(), "Last hidden_dim of output at initial time step must be same as Final Backward Hidden state of final layer in case of Multi layer bi-directional RNN"
```
![shape figure](/assets/images/rnn_hidden/multi_bi_shapes.png "Shape Figure")

*That's all folks for now. If you have any doubts, feedback or any issues in the post, please raise an issue [here](https://github.com/graviraja/pytorch-sample-codes/issues)*

### THANK YOU !!!
{: style="color:black; font-size: 100%; text-align: center;"}