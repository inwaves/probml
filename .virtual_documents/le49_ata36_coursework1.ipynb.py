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
mu, sigma = simple_vae(image)

# TODO: should this have some K*noise*sigma?
# Sample 5 Zs with different noise and generate images from them.
z = [mu + torch.randn_like(sigma)*sigma for _ in range(5)]
new_images = [simple_vae.f(z_i) for z_i in z]
fig = plt.figure(figsize=(10, 10))

for i in range(len(new_images)):
    fig.add_subplot(1, 5, i+1)
    plt.imshow(new_images[i].detach().numpy()[0][0])

plt.axis('off')
plt.show()


z_samples = [torch.rand(4) for _ in range(12)]
generated_images = [simple_vae.f(z_i) for z_i in z_samples]

fig = plt.figure(figsize=(10, 10))

for i in range(len(generated_images)):
    fig.add_subplot(1, 12, i+1)
    plt.imshow(generated_images[i].detach().numpy()[0][0])
plt.axis('off')
plt.show()


(img1, _), (img2, _) = mnist[28001], mnist[43876]
img1 = img1.reshape((1, 1, 28, 28))
img2 = img2.reshape((1, 1, 28, 28))
mu1, sigma1 = simple_vae(img1)
mu2, sigma2 = simple_vae(img2)
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
    new_images.append(simple_vae.f(z_i))
new_images.append(img2)

# Plot all images.
fig = plt.figure(figsize=(10, 10))

for i in range(len(new_images)):
    fig.add_subplot(1, 12, i+1)
    plt.imshow(new_images[i].detach().numpy()[0][0])
plt.axis('off')
plt.show()


# Sort this dictionary for ascending likelihood.
sorted_likelihoods = list(sorted(expanded_sampling_vae.likelihoods.items(), key = lambda item: item[1]))
unlikely_images = []
likely_images = []

# Get mnist[i] for the bottom-most i and for the top-most i
for index in sorted_likelihoods[:10]:
    im, _ = mnist[index[0]]
    unlikely_images.append(im[0])
    
for index in sorted_likelihoods[-10:]:
    im, _ = mnist[index[0]]
    likely_images.append(im[0])
    
# Plot 10 unlikely images followed by 
fig = plt.figure(figsize=(10, 10))

for i in range(len(unlikely_images)):
    fig.add_subplot(2, 10, i+1)
    plt.imshow(unlikely_images[i])
    fig.add_subplot(1, 10, i+1)
    plt.imshow(likely_images[i])
    
plt.axis('off')
plt.show()


vae_more_dims = GaussianEncoder(BernoulliImageGenerator(d=20))
optimizer = optim.Adam(vae_more_dims.parameters())
epoch = 0

while epoch < 10:
    # ...
    pass


def get_greyscale(image):
    img = image.detach().numpy()
    return 784-np.count_nonzero(img < 0.2)-np.count_nonzero(img > 0.7)


count_greyscale = [get_greyscale(img) for (img, _) in mnist]
indices = [i for i in range(len(mnist))]
d = dict(zip(indices, count_greyscale))
digit_index_greyscale_counts = list(sorted(d.items(), key=lambda item : item[1]))

blurry_digits = [index for index, greyscale_count in digit_index_greyscale_counts[-12000:]]


training_set = [mnist[index] for index in range(len(mnist)) if index not in blurry_digits]
holdout_set = [mnist[index] for index in range(len(mnist)) if index in blurry_digits]

batched_training_set = torch.utils.data.DataLoader(training_set, batch_size=100)
batched_holdout_set = torch.utils.data.DataLoader(holdout_set, batch_size=100)


with torch.no_grad():
    loglik_lb, loglike_lb_more_dims = 0, 0
    for batch_num, (images, labels) in enumerate(batched_holdout_set):
        loglik_lb += torch.mean(simple_vae.loglik_lb(batch_num, images))
        loglik_lb_more_dims += torch.mean(vae_more_dims.loglik_lb(batch_num, images))
    print(f"Average log-likelihood of VAE with 4D latent space: {loglik_lb/batch_num}\n
        Average log-likelihood of VAE with 20D latent space: {loglik_lb_more_dims/batch_num}")


y = [5, 3, 9, 1]
y_encoded = one_hot_encode(y, 10)
all_images = []

# Sample four different Z~, decode each using four digits.
for _ in range(4):
    z = torch.randn(1,4)
    all_images.append([cvae.f(z, y_i.unsqueeze(1).T).detach().numpy() for y_i in y_encoded])

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
import torchvision

import random


url = 'https://www.cl.cam.ac.uk/teaching/2122/DataSci/data/xkcd.csv'
xkcd = pandas.read_csv(url)

plt.scatter(xkcd.x, xkcd.y)


mnist = torchvision.datasets.MNIST(
    root = 'pytorch-data/',  # where to put the files
    download = True,         # if files aren't here, download them
    train = True,            # whether to import the test or the train subset
    # PyTorch uses PyTorch tensors internally, not numpy arrays, so convert them.
    transform = torchvision.transforms.ToTensor()
)

# Images can be plotted with matplotlib imshow
show = [mnist[i] for i in [59289, 28001, 35508, 43876, 23627, 14028]]
show = torch.stack([img for img,lbl in show])
x = torchvision.utils.make_grid(show, nrow=6, pad_value=1)
plt.imshow(x.numpy().transpose((1,2,0)))
plt.axis('off')
plt.show()

mnist_batched = torch.utils.data.DataLoader(mnist, batch_size=100)


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


class BernoulliImageGenerator(nn.Module):
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
        return (x*torch.log(xr) + (1-x)*torch.log(1-xr)).sum((1,2,3))


class GaussianEncoder(nn.Module):
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

    def forward(self, x):
        μτ = self.g(x)
        μ,τ = μτ[:,:self.d], μτ[:,self.d:]
        return μ, torch.exp(τ/2)

    def loglik_lb(self, x):
        μ,σ = self(x)
        kl = 0.5 * (μ**2 + σ**2 - torch.log(σ**2) - 1).sum(1)
        ε = torch.randn_like(σ)
        ll = self.f.loglik(x, z=μ+σ*ε)
        return ll - kl


simple_vae = GaussianEncoder(BernoulliImageGenerator())
optimizer = optim.Adam(simple_vae.parameters())
epoch = 0

while epoch < 10:
    for batch_num, (images, _) in enumerate(batched_training_set):
        optimizer.zero_grad()
        loglik_lb = torch.mean(simple_vae.loglik_lb(images))
        (-loglik_lb).backward()
        optimizer.step()
    epoch += 1
    print(f"epoch: {epoch}, loglikelihood: {loglik_lb.item():.4}")


vae_more_dims = GaussianEncoder(BernoulliImageGenerator(20))
optimizer = optim.Adam(vae_more_dims.parameters())
epoch = 0

while epoch < 10:
    for batch_num, (images, _) in enumerate(batched_training_set):
        optimizer.zero_grad()
        loglik_lb = torch.mean(vae_more_dims.loglik_lb(images))
        (-loglik_lb).backward()
        optimizer.step()
    epoch += 1
    print(f"epoch: {epoch}, loglikelihood: {loglik_lb.item():.4}")


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
        self.likelihoods = {}

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

        for i in range(x.shape[0]):
            self.likelihoods[i + batch_num*100] = ll[i].detach().numpy()

        # Sum up all likelihoods to find likelihood of dataset.
        return ll - kl


expanded_sampling_vae = GaussianEncoderExpandedSampling(BernoulliImageGeneratorExpandedSampling())

expanded_sampling_vae.load_state_dict(torch.load("expanded_sampling_vae.pt"))


expanded_sampling_vae = GaussianEncoderExpandedSampling(BernoulliImageGeneratorExpandedSampling())
optimizer = optim.Adam(expanded_sampling_vae.parameters())
epoch = 0

while epoch < 5:
    for batch_num, (images, labels) in enumerate(mnist_batched):
        optimizer.zero_grad()
        loglik_lb = torch.mean(expanded_sampling_vae.loglik_lb(batch_num, images))
        (-loglik_lb).backward()
        optimizer.step()
    epoch += 1
    print(f"epoch: {epoch}, loglikelihood: {loglik_lb.item():.4}")
    torch.save(expanded_sampling_vae.state_dict(), 'expanded_sampling_vae.pt')


def one_hot_encode(y, num_classes):
    encoded = torch.zeros(len(y), num_classes)
    for i in range(len(y)):
        encoded[i, y[i]-1] = 1
    return encoded


class BernoulliImageGeneratorWithStyleCapture(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.d = d
        self.f = nn.Sequential(
            nn.Linear(d + 10, 128),      
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

    def forward(self, z, y):
        # Coerce z and y into a shape that we can feed to the network.
        # z is 100x4, y is 100x10

        z = torch.cat([z, y], dim=1)
        
        return self.f(z)

    def loglik(self, x, z, y):
        xr = self(z, y)
        return (x*torch.log(xr) + (1-x)*torch.log(1-xr)).sum((1,2,3))


class GaussianEncoderWithStyleCapture(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.d = decoder.d
        self.f = decoder
        self.g = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1),          # There are now two channels to account for the added label.
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Flatten(1),
            nn.Linear(9216, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.d*2)
        )

    def forward(self, x, y):
        
        # TODO: Coerce x and y into a form that we can input into the network.
        # Encode y as a 1x28x28, then stack it as an extra channel in the image.
        # So updated y is 100x1x28x28.
        transform_labels = nn.Linear(10, 784)
        
        y_transformed = [transform_labels(y_i) for y_i in y]
        y_transformed = torch.tensor([y_i.view(1, 28, 28).detach().numpy() for y_i in y_transformed]) # Reshaped to Bx1x28x28. Do I need to unsqueeze here given y is already batched?
        
        x = torch.cat([x, y_transformed], dim=1)
        
        mu_tau = self.g(x)
        mu, tau = mu_tau[:,:self.d], mu_tau[:,self.d:]
        return mu, torch.exp(tau/2)

    def loglik_lb(self, x, y):
        """x is 100x1x28x28. y is 100x10 (one-hot encoded class labels)"""
        
        mu, sigma = self(x, y)
        
        kl = 0.5 * (mu**2 + sigma**2 - torch.log(sigma**2) - 1).sum(1)
        
        # Sampling from epsilon is equivalent to generating multiple Z~.
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon
        
        ll = self.f.loglik(x, z, y)
        return ll - kl


cvae = GaussianEncoderWithStyleCapture(BernoulliImageGeneratorWithStyleCapture(4))

cvae.load_state_dict(torch.load("cvae.pt"))
