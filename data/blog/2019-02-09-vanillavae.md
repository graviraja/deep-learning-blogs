---
header:
  overlay_color: "#333"
title: "Vanilla Variational Autoencoder (VAE) in Pytorch"
data: 2019-02-09
tags: [machine learning, data science, deep learning, generative, neural network, encoder, variational autoencoder]
excerpt: "Machine Learning, Variational Autoencoder, Data Science"
mathjax: "true"
---

This post is for the intuition of simple Variational Autoencoder(VAE) implementation in pytorch. The full code is available in my github repo: [link](https://github.com/graviraja/pytorch-sample-codes/blob/master/simple_vae.py)

If you don't know about VAE, go through the following links.

* [VAE blog](http://anotherdatum.com/vae.html)
* [VAE blog](http://kvfrans.com/variational-autoencoders-explained/)

## Variational Autoencoder

### Data processing pipeline
Let's begin with importing stuffs

```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import matplotlib.pyplot as plt

    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

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
    BATCH_SIZE = 64     # number of data points in each batch
    N_EPOCHS = 10       # times to run the model on complete data
    INPUT_DIM = 28 * 28 # size of each input
    HIDDEN_DIM = 256    # hidden dimension
    LATENT_DIM = 20     # latent vector dimension
    lr = 1e-3           # learning rate
```

Define the iterator for the training, testing data. 

```python
    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)
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
        def __init__(self, input_dim, hidden_dim, z_dim):
            '''
            Args:
                input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
                hidden_dim: A integer indicating the size of hidden dimension.
                z_dim: A integer indicating the latent dimension.
            '''
            super().__init__()

            self.linear = nn.Linear(input_dim, hidden_dim)
            self.mu = nn.Linear(hidden_dim, z_dim)
            self.var = nn.Linear(hidden_dim, z_dim)

        def forward(self, x):
            # x is of shape [batch_size, input_dim]

            hidden = F.relu(self.linear(x))
            # hidden is of shape [batch_size, hidden_dim]
            z_mu = self.mu(hidden)
            # z_mu is of shape [batch_size, latent_dim]
            z_var = self.var(hidden)
            # z_var is of shape [batch_size, latent_dim]

            return z_mu, z_var
```

Our $$Q(z \vert X)$$ is a 2 layers network, outputting the $$\mu$$ and  $$\Sigma$$ , the latent parameters of distribution.


### Decoder $$P(X \vert z)$$

The decoder takes a sample from the latent dimension and uses that as an input to output X.
We will see how to sample from latent parameters later in the code.

```python
    class Decoder(nn.Module):
        ''' This the decoder part of VAE

        '''
        def __init__(self, z_dim, hidden_dim, output_dim):
            '''
            Args:
                z_dim: A integer indicating the latent size.
                hidden_dim: A integer indicating the size of hidden dimension.
                output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
            '''
            super().__init__()

            self.linear = nn.Linear(z_dim, hidden_dim)
            self.out = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # x is of shape [batch_size, latent_dim]

            hidden = F.relu(self.linear(x))
            # hidden is of shape [batch_size, hidden_dim]

            predicted = torch.sigmoid(self.out(hidden))
            # predicted is of shape [batch_size, output_dim]

            return predicted

```

Now that we have defined the Encoder and Decoder, let's combine them 

```python
    class VAE(nn.Module):
        ''' This the VAE, which takes a encoder and decoder.

        '''
        def __init__(self, enc, dec):
            super().__init__()

            self.enc = enc
            self.dec = dec

        def forward(self, x):
            # encode
            z_mu, z_var = self.enc(x)

            # sample from the distribution having latent parameters z_mu, z_var
            # reparameterize
            std = torch.exp(z_var / 2)
            eps = torch.randn_like(std)
            x_sample = eps.mul(std).add_(z_mu)

            # decode
            predicted = self.dec(x_sample)
            return predicted, z_mu, z_var

```

### Training

Let's create a instance of our VAE model.

```python
    # encoder
    encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

    # decoder
    decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)

    # vae
    model = VAE(encoder, decoder).to(device)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
```

We use Adam optimizer for our model.

#### Loss
VAE consists of two loss functions
* Reconstruction loss
* KL divergence

So the final objective is 

> loss = reconstruction_loss + kl_divergence

Which we need to optimize.

```python
    def train():
        # set the train mode
        model.train()

        # loss of the epoch
        train_loss = 0

        for i, (x, _) in enumerate(train_iterator):
            # reshape the data into [batch_size, 784]
            x = x.view(-1, 28 * 28)
            x = x.to(device)
            
            # update the gradients to zero
            optimizer.zero_grad()

            # forward pass
            x_sample, z_mu, z_var = model(x)

            # reconstruction loss
            recon_loss = F.binary_cross_entropy(x_sample, x, size_average=False)

            # kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

            # total loss
            loss = recon_loss + kl_loss

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
            for i, (x, _) in enumerate(test_iterator):
                # reshape the data
                x = x.view(-1, 28 * 28)
                x = x.to(device)

                # forward pass
                x_sample, z_mu, z_var = model(x)

                # reconstruction loss
                recon_loss = F.binary_cross_entropy(x_sample, x, size_average=False)
                
                # kl divergence loss
                kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)
                
                # total loss
                loss = recon_loss + kl_loss
                test_loss += loss.item()

        return test_loss

```

Train the model for several epochs.

```python

    best_test_loss = float('inf')

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
![loss fig](/assets/images/simplevae/loss.png "Loss Figure")



#### Testing

Sample from the distribution and generate a image

```python
    # sample and generate a image
    z = torch.randn(1, LATENT_DIM).to(device)

    # run only the decoder
    reconstructed_img = model.dec(z)
    img = reconstructed_img.view(28, 28).data

    print(z.shape)
    print(img.shape)

    plt.imshow(img, cmap='gray')
```


Sample generated image:
![loss fig](/assets/images/simplevae/generated_img.png "Generated Figure")

**Note**: Please raise an issue [here](https://github.com/graviraja/pytorch-sample-codes/issues), if you feel anything wrong.

### THANK YOU !!!
{: style="color:black; font-size: 100%; text-align: center;"}