---
header:
  overlay_color: "#333"
title: "Conditional Variational Autoencoder (VAE) in Pytorch"
data: 2019-03-04
tags: [machine learning, data science, deep learning, generative, neural network, encoder, variational autoencoder]
excerpt: "Machine Learning, Variational Autoencoder, Data Science"
mathjax: "true"
---

This post is for the intuition of Conditional Variational Autoencoder(VAE) implementation in pytorch. The full code is available in my github repo: [link](https://github.com/graviraja/pytorch-sample-codes/blob/master/conditional_vae.py)

If you don't know about VAE, go through the following links.

* [VAE blog](http://anotherdatum.com/vae.html)
* [VAE blog](http://kvfrans.com/variational-autoencoders-explained/)

I have written a blog post on simple autoencoder [here](https://graviraja.github.io/vanillavae/). If you haven't gone the post, once go through it.

## Conditional Variational Autoencoder

*Note: The main variation from the previous post is, in the previous post we generated image randomly. Here we can condition for which number we want to generate the image.*

### Data processing pipeline
Let's begin with importing stuffs

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

```

The code can run on gpu (or) cpu, we can use the gpu if available. In the pytorch we can do this with the following code

```python
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

For the implementation of VAE, I am using the MNIST dataset. Pytorch models accepts data in the form of tensors. So we need to convert the data into form of tensors.

We can do this by defining the transforms, which will be applied on the data.

```python
    transforms = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms)

    test_dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transforms
    )
```

MNIST data contains images of size **28 * 28** 

Next, we will define some parameters which will be used by the model.

```python
BATCH_SIZE = 64         # number of data points in each batch
N_EPOCHS = 10           # times to run the model on complete data
INPUT_DIM = 28 * 28     # size of each input
HIDDEN_DIM = 256        # hidden dimension
LATENT_DIM = 75         # latent vector dimension
N_CLASSES = 10          # number of classes in the data
lr = 1e-3               # learning rate
```

Define the iterator for the training, testing data. 

```python
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)
```

Given a class label, we will convert it into one-hot encoding

```python
def idx2onehot(idx, n=N_CLASSES):

    assert idx.shape[1] == 1
    assert torch.max(idx).item() < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)

    return onehot
```

We have done all the required data processing. Let's dig into variational autoencoder.

In VAE there are two networks:
* Encoder   $$Q(z \vert X)$$
* Decoder  $$P(X \vert z)$$

So, let's build our Encoder $$Q(z \vert X)$$

### Encoder $$Q(z \vert X)$$


```python
class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim + n_classes, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        # latent parameters
        mean = self.mu(hidden)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(hidden)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var
```

Our $$Q(z \vert X)$$ is a 2 layers network, outputting the $$\mu$$ and  $$\Sigma$$ , the latent parameters of distribution.


### Decoder $$P(X \vert z)$$

The decoder takes a sample from the latent dimension and uses that as an input to output X.
We will see how to sample from latent parameters later in the code.

```python
class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = F.relu(self.latent_to_hidden(x))
        # x is of shape [batch_size, hidden_dim]
        generated_x = F.sigmoid(self.hidden_to_out(x))
        # x is of shape [batch_size, output_dim]

        return generated_x

```

Now that we have defined the Encoder and Decoder, let's combine them 

```python
class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_classes)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var

```

### Training

Let's create a instance of our VAE model.

```python
# model
model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES)

#optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
```

We use Adam optimizer for our model.

#### Loss
VAE consists of two loss functions
* Reconstruction loss
* KL divergence

So the final objective is 

> loss = reconstruction_loss + kl_divergence

```python
def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    RCL = F.binary_cross_entropy(reconstructed_x, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD
```
Which we need to optimize.

```python

def train():
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, y) in enumerate(train_iterator):
        # reshape the data into [batch_size, 784]
        x = x.view(-1, 28 * 28)
        x = x.to(device)

        # convert y into one-hot encoding
        y = idx2onehot(y.view(-1, 1))
        y = y.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        reconstructed_x, z_mu, z_var = model(x, y)

        # loss
        loss = calculate_loss(x, reconstructed_x, z_mu, z_var)

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    return train_loss
```

Let's define method for evaluation / testing

```python
def test():
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y) in enumerate(test_iterator):
            # reshape the data
            x = x.view(-1, 28 * 28)
            x = x.to(device)

            # convert y into one-hot encoding
            y = idx2onehot(y.view(-1, 1))
            y = y.to(device)

            # forward pass
            reconstructed_x, z_mu, z_var = model(x, y)

            # loss
            loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
            test_loss += loss.item()

    return test_loss

```

Train the model for several epochs.

```python
for e in range(N_EPOCHS):

    train_loss = train()
    test_loss = test()

    train_loss /= len(train_dataset)
    test_loss /= len(test_dataset)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

    if best_test_loss > test_loss:
        best_test_loss = test_loss
        patience_counter = 1
    else:
        patience_counter += 1

    if patience_counter > 3:
        break
```

Train loss and Test loss:
![loss fig](/assets/images/condvae/loss.png "Loss Figure")



#### Testing

Sample from the distribution and generate a image

```python
# create a random latent vector
z = torch.randn(1, LATENT_DIM).to(device)

# pick randomly 1 class, for which we want to generate the data
y = torch.randint(0, N_CLASSES, (1, 1)).to(dtype=torch.long)
print(f'Generating a {y.item()}')

y = idx2onehot(y).to(device, dtype=z.dtype)
z = torch.cat((z, y), dim=1)

reconstructed_img = model.decoder(z)
img = reconstructed_img.view(28, 28).data

plt.figure()
plt.imshow(img, cmap='gray')
plt.show()
```


Sample generated image:
![loss fig](/assets/images/condvae/number.png "Generated number")
![loss fig](/assets/images/condvae/generated_img.png "Generated Figure")

**Note**: Please raise an issue [here](https://github.com/graviraja/pytorch-sample-codes/issues), if you feel anything wrong.  

### THANK YOU !!!
{: style="color:black; font-size: 100%; text-align: center;"}