---
title: "Transformer Explained - Part 1"
date: '2019-03-18'
tags: [machine learning, data science, deep learning, attention, transformers, sequence-to-sequence]
summary: Transformer is also a typical sequence-to-sequence model, which contains a Encoder and Decoder. Here Encoder contains stack of encoder layers (N=6) and Decoder contains stack of decoder layers (N=6), instead of single encoder and decoder.
draft: false
---

In this post, we will look at **The Transformer** - a model that uses attention to learn the dependencies. This post explains the paper [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf).

![transformer overview fig](/static/images/transformer/transformer.gif)

I highly recommend to read the post [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

The above mentioned post clearly explains the step by step process of Transformer. Here we will go through the overview of each component.


# Overview

> Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation.

> Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences. Such attention mechanisms are used in conjunction with a recurrent network.

> In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.

Transformer is also a typical sequence-to-sequence model, which contains a Encoder and Decoder.

Here Encoder contains stack of encoder layers (N=6) and Decoder contains stack of decoder layers (N=6), instead of single encoder and decoder. 

*N=6* is not a magical number, in the paper it is six layers, one can experiment with other arrangements.

![overview fig](/static/images/transformer/overview.png)

Let's look into the components of single encoder and single decoder and how they interact with each other.

# Encoder

![enc overview fig](/static/images/transformer/encoder_overview.png)

As shown in the above figure, the input is passed through a embedding layer.

The abstraction is that embedding size and all the vectors size inside encoder is kept at **512**.

*Note: Only the first encoder receives embedding of word as input. Other encoders receives the previous layers output as input.*

The Encoder contains 2 layers:
- Multi-Head Attention / Self Attention layer
- Feed Forward layer

## Self Attention

Let's consider the following sentence

> The animal didn't cross the street because it was too tired

What does "it" in this sentence refer to? Is it referring to the street or to the animal? It's a simple question to a human, but not as simple to an algorithm.

Self Attention comes into the play!!

When the model is processing the word "it", self-attention allows it to associate "it" with "animal".

Not only this word, as the model processes each word, self-attention allows it to look at other words in the input for clues that can help lead into a better encoding for the word.

If you're familiar with RNNs, think of how maintaining a hidden state allows an RNN to incorporate its representation of previous words/vectors it has processed with the current one it's processing. Self-attention is the method the Transformer uses to bake the "understanding" of other relevant words into the one we’re currently processing.

Let's see how the self-attention is done.


> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

In the paper self attention is defined as **Scaled Dot-Product Attention**

![selfattn overview fig](/static/images/transformer/selfatt.png "Self Attention Overview Figure") 

What are the “query”, “key”, and “value” vectors? 

Using the above mentioned sentence

> The animal didn't cross the street because it was too tired

We can treat "query" as, what does "it" refer to?

"values" as the vectors for rest of the words.

"keys" are also vectors for each word. By multiplying the "query" vector with "key" vector of word,  it gives a result which indicates how much "value" vector we need to consider.

The final embedding of the word is called "output", which is weighted sum of "value" vectors, weights are from the product of "query" and "key" vectors.

$$\text{Attention(Q, K, V) = softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

Each word has an associated *query, key, value* vectors which are created by multiplying the embeddings with matrices $$W^Q, W^K, W^V$$

For example, let the input is $$X = \{x_1, x_2, x_3\}$$, and the corresponding word embeddings are $$\{w_1, w_2, w_3\}$$

Query, Key, Value vectors for each word are $$\{q_1, q_2, q_3\}$$, $$\{k_1, k_2, k_3\}$$, $$\{v_1, v_2, v_3\}$$ respectively.

**Step 1: Calculate a score** 

Say we’re calculating the self-attention for the first word in this example, $$w_1$$. We need to score each word of the input sentence against this word. The score determines how much focus to place on other parts of the input sentence as we encode a word at a certain position.

The score is calculated by taking the dot product of the *query vector* with the *key vector* of the respective word we’re scoring. So if we’re processing the self-attention for the word $$w_1$$, the first score would be the dot product of $$q_1$$ and $$k_1$$. The second score would be the dot product of $$q_1$$ and $$k_2$$ and the third score is dot product of $$q_1$$ and $$k_3$$.

$$\text{scores = } \{q_1.k_1, q_1.k_2, q_1.k_3\}$$

**Step 2: Scale the score**

> We suspect that for large values of $$d_k$$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $$\frac{1}{\sqrt{d_k}}$$

$$\text{scores = } \{\frac{q_1.k_1}{\sqrt{d_k}}, \frac{q_1.k_2}{\sqrt{d_k}}, \frac{q_1.k_3}{\sqrt{d_k}}\}$$

**Step 3: Apply Softmax**

Softmax normalizes the scores so they’re all positive and add up to 1.

$$\text{scores = Softmax} (\frac{q_1.k_1}{\sqrt{d_k}}, \frac{q_1.k_2}{\sqrt{d_k}}, \frac{q_1.k_3}{\sqrt{d_k}})$$

**Step 4: Compute the product**

Multiply each *value vector* by the softmax score. The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words.

$$\text{weighted values = } \text{Softmax} (\frac{q_1.k_1}{\sqrt{d_k}}, \frac{q_1.k_2}{\sqrt{d_k}}, \frac{q_1.k_3}{\sqrt{d_k}}) * (v_1, v_2, v_3)$$

**Step 5: Output Vector**

Sum up the weighted value vectors. This produces the output of the self-attention layer of the word $$w_1$$

$$\text{output vector for word } w_1 = softmax(\frac{q_1.k_1}{\sqrt{d_k}}).v_1 + softmax(\frac{q_1.k_2}{\sqrt{d_k}}).v_2 + softmax(\frac{q_1.k_3}{\sqrt{d_k}}).v_3$$

Similarly when calculating the self-attention for word $$w_2$$, we need to consider query vector $$q_2$$.

Finally,
> $$\text{Attention(Q, K, V) = softmax}(\frac{QK^T}{\sqrt{d_k}})V$$


## Multi-Head Attention

![multi overview fig](/static/images/transformer/multihead.png "Multi Head Attention Overview Figure") 

> Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values **h** times with different, learned linear projections to $$d_k$$, $$d_k$$ and $$d_v$$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $$d_v$$-dimensional output values. These are concatenated and once again projected, resulting in the final value.

**Main advantage is:** 

It gives the attention layer multiple "representation subspaces". With multi-headed attention we have not only one, but multiple sets of Query/Key/Value weight matrices (the Transformer uses eight attention heads, so we end up with eight sets for each encoder/decoder). Each of these sets is randomly initialized. Then, after training, each set is used to project the input embeddings (or vectors from lower encoders/decoders) into a different representation subspace.


## Positional Encoding

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks

Without positional encodings, the sentences "I like dogs more than cats" and "I like cats more than dogs" encode into same thing. In order to inject some information about the relationship between word positions, **positional encodings** are added to the words.

These positional encodings can be done in 2 ways: *Learned, Fixed*.

Experimental results showed that two versions produced nearly identical results. In the paper, they chose *Fixed* version by choosing sinusoidal waves for encoding positions, as they can extrapolate to sequence lengths longer than the ones encountered during training.

$$PE_{(pos, 2i)} = \sin\biggl(\frac{pos}{10000^{2i/d_{model}}}\biggl)$$

$$PE_{(pos, 2i+1)} = \cos\biggl(\frac{pos}{10000^{2i/d_{model}}}\biggl)$$

**Intution:**

Let's 2 sentences, sentence 1 having 4 words and sentence 2 having 6 words. i.e max_words = 6. Let's conider embedding size is 10.

- batch_size = 2
- max_time = 6
- embedding_size = 10

> input = [2, 6, 10] 

let's initialize all the embeddings to zeros. 

Words at position = 1 across all the batch, have same positional embedding. (This varies in general because word embeddings are not zero)

Words at different positions in the single sentence have different values.

We will look more into this, in the implementation part.

## Position-wise Feed-Forward Layer

The second layer present in Encoder is a position-wise feed forward layer.

A Feed Forward Network is applied to each position separately and identically, containing 1 hidden layer with ReLU activation

$$\text{FFN(x) = } max(0, xW_1 + b_1)W_2 + b_2$$

> While the linear transformations are the same across different positions, they use different parameters from layer to layer.

The dimensionality of input and output is $$d_{model}$$ = 512, and the inner-layer has dimensionality $$d_{ff}$$ = 2048.

## Residual Connections

![residual fig](/static/images/transformer/residual.png "Residual Figure") 

> We employ a residual connection around each of the two sub-layers, followed by layer normalization. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself

Residual connections are the same thing as **skip connections**. They are used to allow gradients to flow through a network directly, without passing through non-linear activation functions.

# Decoder

![dec overview fig](/static/images/transformer/decoder_overview.png "Decoder Overview Figure") 

All the layers are same as Encoder. There is one extra layer present in Decoder which is an **Encoder-Decoder Attention Layer**.

## Encoder-Decoder Attention Layer

The Encoder-Decoder Attention Layer pays attention to the encoder input, while decoding a output.

The "Query" vector is from Decoder self-attention layer. Where as the "Key" and "Value" vectors are from Encoder output. 

## Decoder Self-Attention Layer

As we have already seen how self-attention works, a key note is while decoding a word $$w_t$$, the decoder is not aware of words $$w_{>t}$$, the decoder only knows about words $$w_{<=t}$$. 

How can we perform self-attention then ?

Simple solution is to mask the words $$w_{>t}$$, so that output vector is calculated only from words $$w_{<=t}$$

## Output Layer

![output fig](/static/images/transformer/output.png "Output Figure") 

The output from each decoder input, is passed through a linear layer of output size of target vocabulary. 

Then a Softmax Layer is applied for the output of linear layer, and the word corresponding to high score is taken as output.

# Training

Cross-Entropy is used for calculating the error between target sentence and predicted sentence.

Backward pass is applied and weights of the encoder and decoder are updated w.r.t loss.

*Note : Hope you enjoyed the post* If there is anything wrong/ issue / doubt, please raise an issue [here](https://github.com/graviraja/seq2seq/issues)

### THANK YOU !!!
{: style="color:black; font-size: 100%; text-align: center;"}