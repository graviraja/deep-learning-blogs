---
header:
  overlay_color: "#333"
title: "Transformer Explained - Part 2"
data: 2019-03-22
tags: [machine learning, data science, deep learning, attention, transformers, sequence-to-sequence]
excerpt: "Machine Learning, Deep Learning, Attention"
mathjax: "true"
---

In this post, we will look at implementation of **The Transformer** - a model that uses attention to learn the dependencies. This post explains the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf).

If you are not aware what transformer is, read my previous post about transformer [here](https://graviraja.github.io/transformer/)

The full code is available in my github repo: [link](https://github.com/graviraja/seq2seq/blob/master/transformer.py)

Alternately you can view in colab [here](https://colab.research.google.com/drive/1695mi3IBaubysLCn6SwoG8LCXK4fb8aW)

This post is divided into following sections:
- Imports
- Preparing the data
- Self-Attention Layer
- Positionwise Feedforward Layer
- Positional Encoding
- Encoder Layer
- Encoder
- Decoder Layer
- Decoder
- Transformer
- Training 
- Evaluation
- Running the Model

We will be using PyTorch for coding purposes.

## Imports

The necessary imports for running the code

```python
import os
import math
import time
import spacy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
```

Set the seed value to have deterministic results

```python
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```

## Preparing the data

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

We will set *batch_first=True* argument, to return sentences in batch major format.

```python
SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
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
![data fig](/assets/images/transformer/len_data.png "Data Figure")

Let's see an example from training data

```python
print(f"src: {vars(train_data.examples[0])['src']}")
print(f"trg: {vars(train_data.examples[0])['trg']}")
```
![sample fig](/assets/images/transformer/sample.png "Sample Figure")

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
![vocab fig](/assets/images/transformer/vocab.png "Vocab Figure")

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

## Self-Attention Layer

We will implement the Multi-Head Attention Layer

![multi overview fig](/assets/images/transformer/multihead.png "Multi Head Attention Overview Figure"){: .align-center}

> $$\text{Attention(Q, K, V) = softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

In case of Encoder: Query, Key, Value will be the encoder input

In case of Decoder: Query will be decoder input, Key and Value will be encoder output.

```python
class SelfAttention(nn.Module):
    '''This class implements the Multi-Head Attention.

    Args:
        hid_dim: A integer indicating the hidden dimension.
        n_heads: A integer indicating the number of self attention heads.
        dropout: A float indicating the amount of dropout.
        device: A device to use.
    '''
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0, "Number of heads must be a factor of model dimension"
        # in paper, hid_dim = 512, n_heads = 8

        # query, key, value weight matrices
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        # linear layer to be applied after concating the attention head outputs.
        self.fc = nn.Linear(hid_dim, hid_dim)

        # scale factor to be applied in calculation of self-attention. This is the sqrt of dimension of key vector.
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)
    
    def forward(self, query, key, value, mask=None):
        # query => [batch_size, sent_len, hidden_dim]
        # key => [batch_size, sent_len, hidden_dim]
        # value => [batch_size, sent_len, hidden_dim]

        batch_size = query.shape[0]
        hidden_dim = query.shape[2]
        assert self.hid_dim == hidden_dim, "Hidden dimensions must match"
        
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        # Q, K, V => [batch_size, sent_len, hidden_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        # Q, K, V => [batch_size, n_heads, sent_len, hid_dim//n_heads]

        # z = softmax[(Q.K)/sqrt(q_dim)].V
        # Q => [batch_size, n_heads, sent_len, hid_dim//n_heads]
        # K => [batch_size, n_heads, hid_dim//n_heads, sent_len]
        # Q.K => [batch_size, n_heads, sent_len, sent_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy => [batch_size, n_heads, sent_len, sent_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        # attention => [batch_size, n_heads, sent_len, sent_len]

        x = torch.matmul(attention, V)
        # x => [batch_size, n_heads, sent_len, hid_dim // n_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x => [batch_size, sent_len, n_heads, hid_dim // n_heads]

        # combine all heads
        x = x.view(batch_size, -1, self.hid_dim)

        x = self.fc(x)
        # x => [batch_size, sent_len, hid_dim]
        # x is the output of the Multi-Head Attention Layer
        return x
```

## Positionwise Feedforward Layer

The second layer present in Encoder is a position-wise feed forward layer.

A Feed Forward Network is applied to each position separately and identically, containing 1 hidden layer with ReLU activation

> $$\text{FFN(x) = } max(0, xW_1 + b_1)W_2 + b_2$$

We will use Convolution layer with kernel size 1.

```python
class PositionwiseFeedforward(nn.Module):
    '''This class implements the Position Wise Feed forward Layer.

    This will be applied after the multi-head attention layer.

    Args:
        hid_dim: A integer indicating the hidden dimension of model.
        pf_dim: A integer indicating the position wise feed forward layer hidden dimension.
        dropout: A float indicating the amount of dropout.
    '''
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim    # 2048 in paper

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x => [batch_size, sent_len, hidden_dim]

        x = x.permute(0, 2, 1)
        # x => [batch_size, hidden_dim, sent_len]

        x = self.dropout(F.relu(self.fc_1(x)))
        # x => [batch_size, pf_dim, sent_len]

        x = self.fc_2(x)
        # x => [batch_size, hidden_dim, sent_len]

        x = x.permute(0, 2, 1)
        # x => [batch_size, sent_len, hidden_dim]

        return x
```

## Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks

> $$PE_{(pos, 2i)} = \sin\biggl(\frac{pos}{10000^{2i/d_{model}}}\biggl)$$

> $$PE_{(pos, 2i+1)} = \cos\biggl(\frac{pos}{10000^{2i/d_{model}}}\biggl)$$

```python
class PositionalEncoding(nn.Module):
    '''Implement the PE function.
    
    Args:
        d_model: A integer indicating the hidden dimension of model.
        dropout: A float indicating the amount of dropout.
        max_len: A integer indicating the maximum number of positions for positional encoding.
    '''
    def __init__(self, d_model, dropout, device, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x => [batch_size, seq_len, hidden_dim]

        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
```

## Encoder Layer

![enc overview fig](/assets/images/transformer/encoder_overview.png "Encoder Overview Figure"){: .align-center}

This section contains the implementation of Single Encoder Block.

Each encoder block contains layers:
- self-attention layer followed by layer normalization.
- positionwise feed-forward layer followed by layer normalization.

We have already how to implement SelfAttention Layer and PostionwiseFeedfoward Layer. We will use those and implement the EncoderLayer in the following way:

```python
class EncoderLayer(nn.Module):
    '''This is the single encoding layer module.

    '''
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.ln = nn.LayerNorm(hid_dim)
        self.do = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src => [batch_size, sent_len, hid_dim]
        # src_mask => [batch_size, sent_len]

        # apply the self attention layer for the src, then add the src(residual), and then apply layer normalization
        src = self.ln(src + self.do(self.sa(src, src, src, src_mask)))

        # apply the self positionwise_feedforward layer for the src, then add the src(residual), and then apply layer normalization
        src = self.ln(src + self.do(self.pf(src)))
        return src
```

## Encoder

Encoder contains multiple EncoderLayers. **N=6** given in paper. 

We will convert the src input into embeddings using a Embedding layer, then add the positional to the embeddings which is then passed to initial EncoderLayer.

The output from first EncoderLayer is passed as input to the next EncoderLayer and so on..

```python
class Encoder(nn.Module):
    '''This is the complete Encoder Module.

    It stacks multiple Encoderlayers on top of each other.

    Args:
        input_dim: A integer indicating the input vocab size.
        hid_dim: A integer indicating the hidden dimension of the model.
        n_layers: A integer indicating the number of encoder layers in the encoder.
        n_heads: A integer indicating the number of self attention heads.
        pf_dim: A integer indicating the hidden dimension of positionwise feedforward layer.
        encoder_layer: EncoderLayer class.
        self_attention: SelfAttention Layer class.
        positionwise_feedforward: PositionwiseFeedforward Layer class.
        positional_encoding: A Positional Encoding Instance.
        dropout: A float indicating the amount of dropout.
        device: A device to use.
    '''
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, encoder_layer, self_attention, positionwise_feedforward, positional_encoding, dropout, device):
        super().__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.encoder_layer = encoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.positional_encoding = positional_encoding
        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)        # alternate way of positional encoding

        # Encoder Layers
        self.layers = nn.ModuleList([encoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src => [batch_size, sent_len]
        # src_mask => [batch_size, 1, 1, sent_len]

        src = self.dropout((self.tok_embedding(src) * self.scale))
        src = self.positional_encoding(src)
        # src => [batch_size, sent_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        return src
```

## Decoder Layer

![dec overview fig](/assets/images/transformer/decoder_overview.png "Decoder Overview Figure"){: .align-center}

All the layers are same as EncoderLayer. There is one extra layer present in DecoderLayer which is an **Encoder-Decoder Attention Layer**.

The Encoder-Decoder Attention Layer pays attention to the encoder input, while decoding a output.

The "Query" vector is from Decoder self-attention layer. Where as the "Key" and "Value" vectors are from Encoder output. 

```python
class DecoderLayer(nn.Module):
    '''This is the single Decoder Layer Module.

    Args:
        hid_dim: A integer indicating the hidden dimension of the model.
        n_heads: A integer indicating the number of self attention heads.
        pf_dim: A integer indicating the hidden dimension of positionwise feedforward layer.
        self_attention: SelfAttention class
        positionwise_feedforward: PositionwiseFeedforward Class.
        dropout: A float indicating the amount of dropout.
        device: A device to use.
    '''
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.ln = nn.LayerNorm(hid_dim)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        # trg => [batch_size, trg_len, hid_dim]
        # src => [batch_size, src_len, hid_dim]
        # trg_mask => [batch_size, 1, trg_len, trg_len]
        # src_maks => [batch_size, 1, 1, src_len]

        # self attention is calculated with the target
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        # encoder attention is calculated with src as key, values and trg as query.
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        # positionwise feed forward layer of the decoder
        trg = self.ln(trg + self.do(self.pf(trg)))

        # trg => [batch_size, trg_len, batch_size]
        return trg
```

## Decoder

Decoder contains multiple DecoderLayers. **N=6** given in paper. 

We will convert the trg input into embeddings using a Embedding layer, then add the positional to the embeddings which is then passed to initial DecoderLayer.

The output from first DecoderLayer is passed as input to the next DecoderLayer and so on..

```python
class Decoder(nn.Module):
    '''This is the complete Decoder Module.

    It stacks multiple Decoderlayers on top of each other.

    Args:
        output_dim: A integer indicating the output vocab size.
        hid_dim: A integer indicating the hidden dimension of the model.
        n_layers: A integer indicating the number of encoder layers in the encoder.
        n_heads: A integer indicating the number of self attention heads.
        pf_dim: A integer indicating the hidden dimension of positionwise feedforward layer.
        decoder_layer: DecoderLayer class.
        self_attention: SelfAttention Layer class.
        positional_encoding: A Postional Encoding class.
        positionwise_feedforward: PositionwiseFeedforward Layer class.
        dropout: A float indicating the amount of dropout.
        device: A device to use.
    '''
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention, positionwise_feedforward, positional_encoding, dropout, device):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.positional_encoding = positional_encoding
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)        # alternate way of positional encoding

        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device) for _ in range(n_layers)])
        self.fc = nn.Linear(hid_dim, output_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, src, trg_mask, src_mask):
        # trg => [batch_size, trg_len]
        # src => [batch_size, src_len, hidden_dim]
        # trg_mask => [batch_size, 1, trg_len, trg_len]
        # src_mask => [batch_size, 1, 1, src_len]

        trg = self.do((self.tok_embedding(trg)) * self.scale)
        trg = self.positional_encoding(trg)
        # trg => [batch_size, trg_len, hid_dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        trg = self.fc(trg)
        # trg => [batch_size, trg_len, output_dim]
        return trg
```
## Transformer

This Transformer class encapsulates the Encoder and the Decoder.

![transformer overview fig](/assets/images/transformer/transformer.gif "Transformer Overview Figure"){: .align-center}

We will send the src sentence through the encoder, and obtain the encoded src sentence.

We will send the trg sentence and the encoded src sentence to the decoder, to get the target sentence.

```python
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def make_masks(self, src, trg):
        # src => [batch_size, src_len]
        # trg => [batch_size, trg_len]

        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3)

        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8, device=self.device))

        trg_mask = trg_pad_mask & trg_sub_mask

        return src_mask, trg_mask

    def forward(self, src, trg):
        # src => [batch_size, src_len]
        # trg => [batch_size, trg_len]

        src_mask, trg_mask = self.make_masks(src, trg)

        enc_src = self.encoder(src, src_mask)
        # enc_src => [batch_size, sent_len, hid_dim]

        out = self.decoder(trg, enc_src, trg_mask, src_mask)
        # out => [batch_size, trg_len, output_dim]

        return out
```

## Training

Let's Create the Encoder, Decoder, Transformer 

```python
input_dim = len(SRC.vocab)
output_dim = len(TRG.vocab)
hid_dim = 512
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1
pad_idx = SRC.vocab.stoi['<pad>']

PE = PositionalEncoding(hid_dim, dropout, device)
enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, PE, dropout, device)
dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, PE, dropout, device)
model = Transformer(enc, dec, pad_idx, device).to(device)
```

This is quite a big model. Let's see the number of parameters present in the model.

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"The model has {count_parameters(model) } trainable parameters")

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
```
![param fig](/assets/images/transformer/num_param.png "Param Figure")

In the paper, a modified Adam Optimizer is defined.

We used the Adam optimizer with β1 = 0.9, β2 = 0.98 and  = $$10^{−9}$$
. We varied the learning rate over the course of training, according to the formula:

$$l_{rate} = d^{−0.5}_{model} · min(step\_num^{−0.5}, step\_num · warmup\_steps^{−1.5})$$

```python
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
```

Let's create an instance of optimizer.

```python
optimizer = NoamOpt(hid_dim, 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
```

Next, we define our loss function. The `CrossEntropyLoss` function calculates both the log softmax as well as the negative log-likelihood of our predictions. 

Our loss function calculates the average loss per token, however by passing the index of the `<pad>` token as the `ignore_index` argument we ignore the loss whenever the target token is a padding token. 

```python
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
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.optimizer.zero_grad()

        output = model(src, trg[:,:-1])
                
        #output = [batch size, trg sent len - 1, output dim]
        #trg = [batch size, trg sent len]
            
        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg sent len - 1, output dim]
        #trg = [batch size * trg sent len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
```

## Evaluation

Our evaluation loop is similar to our training loop, however as we aren't updating any parameters we don't need to pass an optimizer or a clip value.

We must remember to set the model to evaluation mode with `model.eval()`. This will turn off dropout (and batch normalization, if used).

We use the `with torch.no_grad()` block to ensure no gradients are calculated within the block. This reduces memory consumption and speeds things up. 

```python
def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg[:,:-1])
            
            #output = [batch size, trg sent len - 1, output dim]
            #trg = [batch size, trg sent len]
            
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg sent len - 1, output dim]
            #trg = [batch size * trg sent len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
```

Let's see how much time each epoch will take to train and evaluate.

```python
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
```

## Running the Model

Let's run the model for 10 epochs. Save the model when there is less loss on validation data.

```python
N_EPOCHS = 10
CLIP = 1
SAVE_DIR = '.'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'transformer-seq2seq.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f'| Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')
```
![train_loss fig](/assets/images/transformer/trainloss.png "Loss Figure")


Load the model to test on test data.

```python
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
```
![test_loss fig](/assets/images/transformer/testloss.png "Loss Figure")


That's all for now. I hope you enjoyed the post.

*Note :* If there is anything wrong/ issue / doubt, please raise an issue [here](https://github.com/graviraja/seq2seq/issues)

### THANK YOU !!!
{: style="color:black; font-size: 100%; text-align: center;"}