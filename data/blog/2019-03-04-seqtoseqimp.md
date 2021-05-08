---
header:
  overlay_color: "#333"
title: "Language Translation using Seq2Seq model in Pytorch"
data: 2019-03-04
tags: [machine learning, data science, deep learning, neural network, sequence to sequence, encoder decoder]
excerpt: "Deep Learning, Sequence to Sequence, Data Science"
mathjax: "true"
---

This post is about the implementation of Language Translation (German -> English) using a Sequence to Sequence Model.

If you don't know about sequence-to-sequence models, refer to my previous post [here](https://graviraja.github.io/seqtoseq/)

We will use PyTorch for writing our model, and also TorchText to do all the pre-processing of the data.

We'll be using [Multi30k dataset](https://github.com/multi30k/dataset). This is a dataset with ~30,000 parallel English, German and French sentences.

*Let's get into code...* The full code is available in my github repo: [link](https://github.com/graviraja/seq2seq/blob/master/simple_seq2seq.py)

### Imports

First things first, let's import all the necessary libraries

```python
import os
import math
import random
import spacy

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

```

Set the seed value to have deterministic results

```python
SEED = 5
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

### Preparing the data

We will be using spacy for tokenizing the data

```python
# for tokenizing the english sentences
spacy_en = spacy.load('en')
# for tokenizing the german sentences
spacy_de = spacy.load('de')
```

Next, we create the tokenizer functions. These can be passed to TorchText `Field` and will take in the sentence as a string and return the sentence as a list of tokens.


```python
def tokenize_de(text):
    # tokenizes the german text into a list of strings(tokens) and reverse it
    # we are reversing the input sentences, as it is observed 
    # by reversing the inputs we will get better results
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]     # list[::-1] used to reverse the list


def tokenize_en(text):
    # tokenizes the english text into a list of strings(tokens)
    return [tok.text for tok in spacy_en.tokenizer(text)]
```

Torchtext's Field handle how the data should be processed. For more refer [here](https://github.com/pytorch/text)

Use the tokenize_de, tokenize_en for tokenization of german and english sentences.

German is the src (input language), English is the trg (output language)

We will create `Fields`, German being the SRC (source) field and English being the TRG (target) field. 

Append the `<sos>` (start of sentence), `<eos>` (end of sentence) tokens to all sentences. This can be done simply by specifing the *init_token*, *eos_token* arguments in the *Field*

We can also configure which tokenization to use by specifying the *tokenize* argument.

```python
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
```

Since we decide German is the input language and English as the target language, we need to define that in *exts*

`exts` specifies which languages to use as source and target, source goes first.
`fields` define which data processing to apply for source and target.

```python
train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
print('Loaded data...')
```

Let's check the amount of data in training, validation, test datasets.

```python
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")
```
![exm fig](/assets/images/seq2seq/num_examples.png "Examples Figure")

Let's see an example from training data

*Note : source sentence is reversed.*

```python
print(f"src: {vars(train_data.examples[0])['src']}")
print(f"trg: {vars(train_data.examples[0])['trg']}")
```
![sam_exm fig](/assets/images/seq2seq/sample_example.png "Example Figure")

### Building the vocabulary

Since the source and target are in different languages, we need to build the vocabulary for the both languages.

With Torchtext's `Field` that is extremely simple. we don't need to worry about creating dicts, mapping word to index, mapping index to word, counting the words etc. All these things are done by the `Field` for us.

We can define the minimum frequency of the words by specifying the attribute `min_freq` in `build_vocab` method of `Field`. Tokens that appear less the `min_freq` are converted into an `<unk>` (unknown) token.

*Note : We will use only training data for creating the vocabulary*

```python
# build the vocab
# consider words which are having atleast min_freq.
# words having less than min_freq will be replaced by <unk> token
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)
print('Vocab builded...')
```

Let's see the size of our src and trg vocabulary

```python
print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")
```
![num_tok fig](/assets/images/seq2seq/num_tokens.png "Tokens Figure")

```python
# define batch size
BATCH_SIZE = 32
```

Let's create the iterators for our data.

These can be iterated on to return a batch of data which will have a src attribute (the PyTorch tensors containing a batch of numericalized source sentences) and a trg attribute (the PyTorch tensors containing a batch of numericalized target sentences).

We also need to replace the words by it's indexes, since any model takes only numbers as input using the `vocabulary`.

We use a `BucketIterator` instead of the standard `Iterator` as it creates batches in such a way that it minimizes the amount of padding in both the source and target sentences.

This can be done as following:

```python
# use gpu if available, else use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create data iterators for the data
# padding all the sentences to same length, replacing words by its index,
# bucketing (minimizes the amount of padding by grouping similar length sentences)
train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device)

```

## Building the Model

Our model mainly contains three parts. The encoder, the decoder and a seq2seq model that encapsulates the encoder and decoder.

### Encoder

As we have already seen, the encoder is a RNN that takes a input sentence and produces a context vector.

We will use LSTM in the encoder, a 2 layer LSTM.

![enc fig](/assets/images/seq2seq/encoder.png "Encoder Figure")

For a multi-layer RNN, the input sentence, $$X$$, goes into the first (bottom) layer of the RNN and hidden states, $$H=\{h_1, h_2, ..., h_T\}$$, output by this layer are used as inputs to the RNN in the layer above. Thus, representing each layer with a superscript, the hidden states in the first layer are given by:

$$h_t^1 = \text{EncoderRNN}^1(x_t, h_{t-1}^1)$$

The hidden states in the second layer are given by:

$$h_t^2 = \text{EncoderRNN}^2(h_t^1, h_{t-1}^2)$$

Without going into too much detail about LSTMs (see [this](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) blog post if you want to learn more about them), all we need to know is that they're a type of RNN which instead of just taking in a hidden state and returning a new hidden state per time-step, also take in and return a *cell state*, $$c_t$$, per time-step.

$$\begin{align*}
h_t &= \text{RNN}(x_t, h_{t-1})\\
(h_t, c_t) &= \text{LSTM}(x_t, (h_{t-1}, c_{t-1}))
\end{align*}$$


You can just think of $$c_t$$ as another type of hidden state. Similar to $$h_0^l$$, $$c_0^l$$ will be initialized to a tensor of all zeros. Also, our context vector will now be both the final hidden state and the final cell state, i.e. $$z^l = (h_T^l, c_T^l)$$.

Extending our multi-layer equations to LSTMs, we get:

$$\begin{align*}
(h_t^1, c_t^1) &= \text{EncoderLSTM}^1(x_t, (h_{t-1}^1, c_{t-1}^1))\\
(h_t^2, c_t^2) &= \text{EncoderLSTM}^2(h_t^1, (h_{t-1}^2, c_{t-1}^2))
\end{align*}$$

Note how only our hidden state from the first layer is passed as input to the second layer, and not the cell state.

One thing to note is that the dropout argument to the LSTM is how much dropout to apply between the layers of a multi-layer RNN, i.e. between the hidden states output from layer `l`  and those same hidden states being used for the input of layer  `l+1`.

In the `forward` method, we pass in the source sentence, $$X$$, which is converted into dense vectors using the `embedding` layer, and then dropout is applied. These embeddings are then passed into the RNN. As we pass a whole sequence to the RNN, it will automatically do the recurrent calculation of the hidden states over the whole sequence for us! 

The RNN returns: `outputs` (the top-layer hidden state for each time-step), `hidden` (the final hidden state for each layer,  $$h_T$$, stacked on top of each other) and `cell` (the final cell state for each layer,  $$c_T$$ , stacked on top of each other).

As we only need the final hidden and cell states (to make our context vector), forward only returns hidden and cell.

So our encoder code looks like this: 

```python
class Encoder(nn.Module):
    ''' Sequence to sequence networks consists of Encoder and Decoder modules.
    This class contains the implementation of Encoder module.
    Args:
        input_dim: A integer indicating the size of input dimension.
        emb_dim: A integer indicating the size of embeddings.
        hidden_dim: A integer indicating the hidden dimension of RNN layers.
        n_layers: A integer indicating the number of layers.
        dropout: A float indicating dropout.
    '''
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout)  # default is time major
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src is of shape [sentence_length, batch_size], it is time major

        # embedded is of shape [sentence_length, batch_size, embedding_size]
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        # inputs to the rnn is input, (h, c); if hidden, cell states are not passed means default initializes to zero.
        # input is of shape [sequence_length, batch_size, input_size]
        # hidden is of shape [num_layers * num_directions, batch_size, hidden_size]
        # cell is of shape [num_layers * num_directions, batch_size, hidden_size]
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs are always from the top hidden layer, if bidirectional outputs are concatenated.
        # outputs shape [sequence_length, batch_size, hidden_dim * num_directions]
        return hidden, cell
```

### Decoder

The decoder is a RNN that takes the context vector and produces one word at a time.

We will use LSTM in the decoder, a 2 layer LSTM.

![dec fig](/assets/images/seq2seq/decoder.png "Decoder Figure")

The `Decoder` class does decoding, one step at a time. The first layer of the decoder will receive a hidden and cell state from the previous time step, $$(s_{t-1}^1, c_{t-1}^1)$$, and feed it through the LSTM with the current token, $$y_t$$, to produce a new hidden and cell state $$(s_t^1, c_t^1)$$. The subsequent layers will use the hidden state from the layer below, $$s_t^{l-1}$$, and previous hidden and cell states from the same layer, $$(s_{t-1}^l, c_{t-1}^l)$$. This can be seen as:

$$\begin{align*}
(s_t^1, c_t^1) &= \text{DecoderLSTM}^1(y_t, (s_{t-1}^1, c_{t-1}^1))\\
(s_t^2, c_t^2) &= \text{DecoderLSTM}^2(s_t^1, (s_{t-1}^2, c_{t-1}^2))
\end{align*}$$

*Note :* The initial hidden and cell states to our decoder are our context vectors, which are the final hidden and cell states of our encoder from the same layer. i.e $$(s_0^l, c_0^l) = z^l = (h_T^l, c_T^l)$$

We then pass the hidden state from the top layer of the RNN, $$s_t^l$$, through a linear layer $$f$$, to make a prediction of what the next token in the target (output) sequence should be $$\hat{y}_{t+1}$$ 

$$\begin{align*}
\hat{y}_{t+1} = f(s_t^l)
\end{align*}$$

In the `forward` method, we pass in the input tokens along with the previous layers hidden states and cell states. We convert the input tokens into dense vectors using the `embedding` layer, and then dropout is applied. These embeddings are then passed into the RNN. This produces a new hidden and cell state. We then pass the `hidden` state through the linear layer to make our `prediction`. We then return the `prediction`, new `hidden` state and the `cell` state.

So our decoder code looks like this: 

```python
class Decoder(nn.Module):
    ''' This class contains the implementation of Decoder Module.
    Args:
        embedding_dim: A integer indicating the embedding size.
        output_dim: A integer indicating the size of output dimension.
        hidden_dim: A integer indicating the hidden size of rnn.
        n_layers: A integer indicating the number of layers in rnn.
        dropout: A float indicating the dropout.
    '''
    def __init__(self, embedding_dim, output_dim, hidden_dim, n_layers, dropout):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input is of shape [batch_size]
        # hidden is of shape [n_layer * num_directions, batch_size, hidden_size]
        # cell is of shape [n_layer * num_directions, batch_size, hidden_size]

        input = input.unsqueeze(0)
        # input shape is [1, batch_size]. reshape is needed rnn expects a rank 3 tensors as input.
        # so reshaping to [1, batch_size] means a batch of batch_size each containing 1 index.

        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        # embedded is of shape [1, batch_size, embedding_dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # generally output shape is [sequence_len, batch_size, hidden_dim * num_directions]
        # generally hidden shape is [num_layers * num_directions, batch_size, hidden_dim]
        # generally cell shape is [num_layers * num_directions, batch_size, hidden_dim]

        # sequence_len and num_directions will always be 1 in the decoder.
        # output shape is [1, batch_size, hidden_dim]
        # hidden shape is [num_layers, batch_size, hidden_dim]
        # cell shape is [num_layers, batch_size, hidden_dim]

        predicted = self.linear(output.squeeze(0))  # linear expects as rank 2 tensor as input
        # predicted shape is [batch_size, output_dim]

        return predicted, hidden, cell
```

### Seq2Seq

For the final part, we'll implement the seq2seq model which encapsulates the encoder and decoder modules. This contains:
* receiving the input/source sentence
* using the encoder to produce the context vectors
* using the decoder to produce the predicted output/target sentence

Complete model: 

![model fig](/assets/images/seq2seq/fullmodel.png "Model Figure")

For this implementation, we have to ensure that the number of layers and the hidden (and cell) dimensions are equal in the `Encoder` and `Decoder`. This is not always the case, you do not necessarily need the same number of layers or the same hidden dimension sizes in a sequence-to-sequence model. However, if you do something like having a different number of layers you will need to make decisions about how this is handled. For example, if your encoder has 2 layers and your decoder only has 1, how is this handled? Do you average the two context vectors output by the decoder? Do you pass both through a linear layer? Do you only use the context vector from the highest layer? etc.

Our `forward` method takes the source sentence, target sentence and a teacher-forcing ratio. The teacher forcing ratio is used when training our model. When decoding, at each time-step we will predict what the next token in the target sequence will be from the previous tokens decoded, $$\hat{y}_{t+1}=f(s_t^L)$$. With probability equal to the teaching forcing ratio (`teacher_forcing_ratio`) we will use the actual ground-truth next token in the sequence as the input to the decoder during the next time-step. However, with probability `1 - teacher_forcing_ratio`, we will use the token that the model predicted as the next input to the model, even if it doesn't match the actual next token in the sequence.  

The first thing we do in the `forward` method is to create an `outputs` tensor that will store all of our predictions, $$\hat{Y}$$.

We then feed the input/source sentence, $$X$$/`src`, into the encoder and receive out final hidden and cell states.

The first input to the decoder is the start of sequence (`<sos>`) token. As our `trg` tensor already has the `<sos>` token appended (all the way back when we defined the `init_token` in our `TRG` field) we get our $$y_1$$ by slicing into it. We know how long our target sentences should be (`max_len`), so we loop that many times. During each iteration of the loop, we:
- pass the input, previous hidden and previous cell states ($$y_t, s_{t-1}, c_{t-1}$$) into the decoder
- receive a prediction, next hidden state and next cell state ($$\hat{y}_{t+1}, s_{t}, c_{t}$$) from the decoder
- place our prediction, $$\hat{y}_{t+1}$$/`output` in our tensor of predictions, $$\hat{Y}$$/`outputs`
- decide if we are going to "teacher force" or not
    - if we do, the next `input` is the ground-truth next token in the sequence, $$y_{t+1}$$/`trg[t]`
    - if we don't, the next `input` is the predicted next token in the sequence, $$\hat{y}_{t+1}$$/`top1`
    
Once we've made all of our predictions, we return our tensor full of predictions, $$\hat{Y}$$/`outputs`.

```python
class Seq2Seq(nn.Module):
    ''' This class contains the implementation of complete sequence to sequence network.
    It uses to encoder to produce the context vectors.
    It uses the decoder to produce the predicted target sentence.
    Args:
        encoder: A Encoder class instance.
        decoder: A Decoder class instance.
    '''
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src is of shape [sequence_len, batch_size]
        # trg is of shape [sequence_len, batch_size]
        # if teacher_forcing_ratio is 0.5 we use ground-truth inputs 50% of time and 50% time we use decoder outputs.

        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # to store the outputs of the decoder
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size)

        # context vector, last hidden and cell state of encoder to initialize the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if use_teacher_force else top1)

        # outputs is of shape [sequence_len, batch_size, output_dim]
        return outputs
```

### Training the model

Now that we have implemented our model, let's train it.

We'll first define all the model parameters, and create the model instance.

```python
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 256   # encoder embedding size
DEC_EMB_DIM = 256   # decoder embedding size (can be different from encoder embedding size)
HID_DIM = 512       # hidden dimension (must be same for encoder & decoder)
N_LAYERS = 2        # number of rnn layers (must be same for encoder & decoder)
ENC_DROPOUT = 0.5   # encoder dropout
DEC_DROPOUT = 0.5   # decoder dropout (can be different from encoder droput)

# encoder
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
# decoder
dec = Decoder(DEC_EMB_DIM, OUTPUT_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
# model
model = Seq2Seq(enc, dec)
```

We define our optimizer, which we use to update our parameters in the training loop. Here, we'll use Adam.

```python
optimizer = optim.Adam(model.parameters())
```

Next, we define our loss function. The `CrossEntropyLoss` function calculates both the log softmax as well as the negative log-likelihood of our predictions. 

Our loss function calculates the average loss per token, however by passing the index of the `<pad>` token as the `ignore_index` argument we ignore the loss whenever the target token is a padding token. 

```python
pad_idx = TRG.vocab.stoi['<pad>']
# loss function calculates the average loss per token
# passing the <pad> token to ignore_idx argument, we will ignore loss whenever the target token is <pad>
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
```

Next, we'll define our training loop. 

First, we'll set the model into "training mode" with `model.train()`. This will turn on dropout (and batch normalization, which we aren't using) and then iterate through our data iterator.

At each iteration:
- get the source and target sentences from the batch, $$X$$ and $$Y$$
- zero the gradients calculated from the last batch
- feed the source and target into the model to get the output, $$\hat{Y}$$
- as the loss function only works on 2d inputs with 1d targets we need to flatten each of them with `.view`
    - we also don't want to measure the loss of the `<sos>` token, hence we slice off the first column of the output and target tensors
- calculate the gradients with `loss.backward()`
- clip the gradients to prevent them from exploding (a common issue in RNNs)
- update the parameters of our model by doing an optimizer step
- sum the loss value to a running total

Finally, we return the loss that is averaged over all batches.

```python
def train(model, iterator, optimizer, criterion, clip):
    ''' Training loop for the model to train.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        optimizer: Optimizer for the model.
        criterion: loss criterion.
        clip: gradient clip value.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.train()
    # loss
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        # trg is of shape [sequence_len, batch_size]
        # output is of shape [sequence_len, batch_size, output_dim]
        output = model(src, trg)

        # loss function works only 2d logits, 1d targets
        # so flatten the trg, output tensors. Ignore the <sos> token
        # trg shape shape should be [(sequence_len - 1) * batch_size]
        # output shape should be [(sequence_len - 1) * batch_size, output_dim]
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

        # backward pass
        loss.backward()

        # clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the parameters
        optimizer.step()

        epoch_loss += loss.item()

    # return the average loss
    return epoch_loss / len(iterator)
```

Our evaluation loop is similar to our training loop, however as we aren't updating any parameters we don't need to pass an optimizer or a clip value.

We must remember to set the model to evaluation mode with `model.eval()`. This will turn off dropout (and batch normalization, if used).

We use the `with torch.no_grad()` block to ensure no gradients are calculated within the block. This reduces memory consumption and speeds things up. 

The iteration loop is similar (without the parameter updates), however we must ensure we turn teacher forcing off for evaluation. This will cause the model to only use it's own predictions to make further predictions within a sentence, which mirrors how it would be used in deployment.

```python
def evaluate(model, iterator, criterion):
    ''' Evaluation loop for the model to evaluate.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        criterion: loss criterion.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.eval()
    # loss
    epoch_loss = 0

    # we don't need to update the model parameters. only forward pass.
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)     # turn off the teacher forcing

            # loss function works only 2d logits, 1d targets
            # so flatten the trg, output tensors. Ignore the <sos> token
            # trg shape shape should be [(sequence_len - 1) * batch_size]
            # output shape should be [(sequence_len - 1) * batch_size, output_dim]
            loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].view(-1))

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)
```

We can finally start training our model!

At each epoch, we'll be checking if our model has achieved the best validation loss so far. If it has, we'll update our best validation loss and save the parameters of our model (called `state_dict` in PyTorch). Then, when we come to test our model, we'll use the saved parameters used to achieve the best validation loss. 

We'll be printing out both the loss and the perplexity at each epoch. It is easier to see a change in perplexity than a change in loss as the numbers are much bigger.

```python
N_EPOCHS = 10           # number of epochs
CLIP = 10               # gradient clip value
SAVE_DIR = 'models'     # directory name to save the models.
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'seq2seq_model.pt')

best_validation_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    if valid_loss < best_validation_loss:
        best_validation_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
```

We'll load the parameters (`state_dict`) that gave our model the best validation loss and run it the model on the test set.

```python
# load the parameters(state_dict) that gave the best validation loss and run the model to test.
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
```

![epoch_loss fig](/assets/images/seq2seq/epoch_loss.png "Loss Figure")

*Note :* If there is anything wrong/ issue / doubt, please raise an issue [here](https://github.com/graviraja/seq2seq/issues)

### References

[Seq2Seq from pytorch documentation](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

[Seq2Seq code by bentrevett](https://github.com/bentrevett/pytorch-seq2seq/)

### THANK YOU !!!
{: style="color:black; font-size: 100%; text-align: center;"}