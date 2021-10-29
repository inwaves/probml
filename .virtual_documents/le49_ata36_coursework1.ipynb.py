mu_x = simple_wiggle.mu(x).detach().numpy()
sigma = simple_wiggle.sigma.item()

plt.plot(xkcd.x, mu_x)
plt.fill_between(xkcd.x, (mu_x-1.96*sigma).reshape(31,), (mu_x+1.96*sigma).reshape(31,), alpha=0.2)


mu_x = heteroscedastic_wiggle.mu(x).detach().numpy()
sigma = heteroscedastic_wiggle.sigma(x).detach().numpy()

plt.plot(xkcd.x, mu_x)
plt.fill_between(xkcd.x, (mu_x-1.96*sigma).reshape(31,), (mu_x+1.96*sigma).reshape(31,), alpha=0.2)


def q(x):
    _, counts = np.unique(x, return_counts=True)
    return counts/x.shape[0]


q_x = q(x)

np.sum(q_x * np.log(q_x))


epochs = np.arange(1, 50000, 100)

plt.plot(epochs, heteroscedastic_training_likelihood)
plt.plot(epochs, simple_training_loglik)


new_images = []

for image in showr:
    
    # Add the original image.
    new_images.append(image.detach().numpy()[0][0])
    
    # Generate 3 reconstructions of the image.
    for i in range(3):
        mu, sigma = simple_vae(image)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon
        new_image = simple_vae.f(z)
        new_images.append(new_image.detach().numpy()[0][0])

cols, rows = 4, 4
fig = plt.figure(figsize=(10, 10))

for i in range(1, cols*rows + 1):
    fig.add_subplot(rows, cols, i)
    plt.imshow(new_images[i-1])
plt.show()


image, _ = mnist[35508]
image = image.reshape((1, 1, 28, 28)) # Must be [Bx1x28x28].
mu, sigma = autoencoder(image)

# TODO: should this have some K*noise*sigma?
# Sample 5 Zs with different noise and generate images from them.
z = [mu + torch.randn_like(sigma)*sigma for _ in range(5)]
new_images = [autoencoder.f(z_i) for z_i in z]
fig = plt.figure(figsize=(10, 10))

for i in range(len(new_images)):
    fig.add_subplot(1, 5, i+1)
    plt.imshow(new_images[i].detach().numpy()[0][0])

plt.axis('off')
plt.show()


z_samples = [torch.rand(4) for _ in range(12)]
generated_images = [autoencoder.f(z_i) for z_i in z_samples]

fig = plt.figure(figsize=(10, 10))

for i in range(len(generated_images)):
    fig.add_subplot(1, 12, i+1)
    plt.imshow(generated_images[i].detach().numpy()[0][0])
plt.axis('off')
plt.show()


(img1, _), (img2, _) = mnist[28001], mnist[43876]
img1 = img1.reshape((1, 1, 28, 28))
img2 = img2.reshape((1, 1, 28, 28))
mu1, sigma1 = autoencoder(img1)
mu2, sigma2 = autoencoder(img2)
z1 = mu1 + torch.randn_like(sigma1)
z2 = mu2 + torch.randn_like(sigma2)

# Generate 10 z* interpolated between z1 and z2.
interpolated_vars = []
for _ in range(10):
    coef = random.random()
    interpolated_vars.append(z1 + coef*(z2-z1))

# Add the first of the original images.
new_images = [img1]
for z_i in interpolated_vars:
    new_images.append(autoencoder.f(z_i))
new_images.append(img2)

# Plot all images.
fig = plt.figure(figsize=(10, 10))

for i in range(len(new_images)):
    fig.add_subplot(1, 12, i+1)
    plt.imshow(new_images[i].detach().numpy()[0][0])
plt.axis('off')
plt.show()


def loglik_lb(self, batch_num, x):
        μ,σ = self(x)
        kl = 0.5 * (μ**2 + σ**2 - torch.log(σ**2) - 1).sum(1)

        ll = 0
        num_samples = 1
        for _ in range(num_samples):
            ε = torch.randn_like(σ)
            ll += self.f.loglik(x, z=μ+σ*ε)
        ll /= num_samples

        # Store the image with the lowest likelihood across all batches.
        if torch.min(ll) < self.min_likelihood[0]:
            self.min_likelihood = torch.min(ll), torch.argmin(ll) + batch_num*100

        # Store the image with the highest likelihood across all batches.
        if torch.max(ll) < self.max_likelihood[0]:
            self.max_likelihood = torch.max(ll), torch.argmax(ll) + batch_num*100

        # Sum up all likelihoods to find likelihood of dataset.
        return ll - kl


likely, _ = mnist[72]
unlikely, _ = mnist[1618]

fig = plt.figure(figsize=(5, 5))

fig.add_subplot(1, 2, 1)
plt.imshow(unlikely[0])
plt.axis('off')

fig.add_subplot(1, 2, 2)
plt.imshow(likely[0])
plt.axis('off')

plt.show()


autoencoder = GaussianEncoder(BernoulliImageGenerator(d=20))
optimizer = optim.Adam(autoencoder.parameters())
epoch = 0

while epoch < 10:
    # ...
    pass


y = [5, 3, 9, 1]
y_encoded = one_hot_encode(y, 10)
all_images = []

# Sample four different Z~, decode each using four digits.
for _ in range(4):
    z = torch.randn(1,4)
    all_images.append([autoencoder.f(z, y_i.unsqueeze(1).T).detach().numpy() for y_i in y_encoded])

# Plot results.
fig = plt.figure(figsize=(10, 10))
for j in range(len(all_images)):
    for i in range(len(all_images[j])):
        fig.add_subplot(len(all_images[j]), len(all_images), 4*j+(i+1))
        plt.imshow(all_images[j][i][0][0])


import pandas
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch

import random


url = 'https://www.cl.cam.ac.uk/teaching/2122/DataSci/data/xkcd.csv'
xkcd = pandas.read_csv(url)

plt.scatter(xkcd.x, xkcd.y)


class Wiggle(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(1,4),
            nn.LeakyReLU(),
            nn.Linear(4,20),
            nn.LeakyReLU(),
            nn.Linear(20,20),
            nn.LeakyReLU(),
            nn.Linear(20,1)
        )
    def forward(self, x):
        return self.f(x)
    
class RWiggle(nn.Module):
    """Y_i ~ N(f(x_i), sigma2)"""
    def __init__(self):
        super().__init__()
        self.mu = Wiggle()
        self.sigmapre = nn.Parameter(torch.tensor(1.0))
    @property
    def sigma(self):
         return torch.nn.functional.softplus(self.sigmapre)
    def forward(self, y, x):
        sigma2 = torch.pow(self.sigma, 2)
        return - 0.5*torch.log(2*np.pi*sigma2) - torch.pow(y - self.mu(x), 2) / (2*sigma2)


x = torch.tensor(xkcd.x, dtype=torch.float)[:, None]
y = torch.tensor(xkcd.y, dtype=torch.float)[:, None]

simple_wiggle = RWiggle()
epoch = 0
optimizer = optim.Adam(simple_wiggle.parameters())

# Store the log-likelihood of the dataset every 100 epochs
# so we can plot it against epochs and compare with the 
# heteroscedastic model.
simple_training_loglik = []

with Interruptable() as check_interrupted:
    while epoch < 50000:
        check_interrupted()
        optimizer.zero_grad()
        loglikelihood = torch.mean(simple_wiggle(y, x))
        (-loglikelihood).backward()
        optimizer.step()
        epoch += 1
        if epoch % 100 == 0:
            simple_training_loglik.append(loglikelihood.item())


class HeteroscedasticRWiggle(nn.Module):
    """Y_i ~ N(f(x_i), f(x_i)^2)"""
    def __init__(self):
        super().__init__()
        self.mu = Wiggle()
        self.sigmapre = Wiggle()
    def sigma(self, x):
         return torch.nn.functional.softplus(self.sigmapre(x))
        
    def forward(self, y, x):
        sigma2 = torch.pow(self.sigma(x), 2)
        return - 0.5*torch.log(2*np.pi*sigma2) - torch.pow(y - self.mu(x), 2) / (2*sigma2)


heteroscedastic_wiggle = HeteroscedasticRWiggle()
epoch = 0
optimizer = optim.Adam(heteroscedastic_wiggle.parameters())
heteroscedastic_training_likelihood = []

with Interruptable() as check_interrupted:
    while epoch < 50000:
        check_interrupted()
        optimizer.zero_grad()
        loglikelihood = torch.mean(heteroscedastic_wiggle(y, x))
        (-loglikelihood).backward()
        optimizer.step()
        epoch += 1
        if epoch % 100 == 0:
            heteroscedastic_training_likelihood.append(loglikelihood.item())


class BernoulliImageGeneratorExpandedSampling(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.d = d
        self.f = nn.Sequential(
            nn.Linear(d, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1728),
            nn.LeakyReLU(),
            nn.Unflatten(1, (12,12,12)), # -> B×12×12×12
            nn.Conv2d(12, 36, 3, 1),     # -> [B×36×10×10]
            nn.LeakyReLU(),
            nn.Flatten(1),               # -> [B×3600]
            nn.Unflatten(1, (4,30,30)),  # -> [B×4×30×30]
            nn.Conv2d(4, 4, 3, 1),       # -> [B×4×28×28]
            nn.LeakyReLU(),
            nn.Conv2d(4, 1, 1, 1),       # -> [B×1×28×28]
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.f(z)

    def loglik(self, x, z):
        xr = self(z)
        return (x*torch.log(xr) + (1-x)*torch.log(1-xr)).sum((1, 2, 3))


class GaussianEncoderExpandedSampling(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.d = decoder.d
        self.f = decoder
        self.g = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.d*2)
        )
        self.max_likelihood = (0, 0)
        self.min_likelihood = (0, 0)

    def forward(self, x):
        μτ = self.g(x)
        μ,τ = μτ[:,:self.d], μτ[:,self.d:]
        return μ, torch.exp(τ/2)

    def loglik_lb(self, batch_num, x):
        μ,σ = self(x)
        kl = 0.5 * (μ**2 + σ**2 - torch.log(σ**2) - 1).sum(1)

        ll = 0
        num_samples = 1
        for _ in range(num_samples):
            ε = torch.randn_like(σ)
            ll += self.f.loglik(x, z=μ+σ*ε)
        ll /= num_samples

        if torch.min(ll) < self.min_likelihood[0]:
            self.min_likelihood = torch.min(ll), torch.argmin(ll) + batch_num*100

        if torch.max(ll) < self.max_likelihood[0]:
            self.max_likelihood = torch.max(ll), torch.argmax(ll) + batch_num*100

        # Sum up all likelihoods to find likelihood of dataset.
        return ll - kl



