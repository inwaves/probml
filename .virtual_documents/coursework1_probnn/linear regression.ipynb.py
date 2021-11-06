import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


prior_a = norm(loc=0, scale=4)
prior_b = norm(loc=2, scale=2)
prior_c = norm(loc=-2, scale=5)


x = np.linspace(-20, 20)
x


fig, ax = plt.subplots(1, 1)
ax.fill(x, prior_a.pdf(x), alpha=0.7)
ax.fill(x, prior_b.pdf(x), alpha=0.5)
ax.fill(x, prior_c.pdf(x), alpha=0.5)


sampled_a, sampled_b, sampled_c = prior_a.rvs(size=1), prior_b.rvs(size=1), prior_c.rvs(size=1)


def compute_y(sampled_a, sampled_b, sampled_c, X):
    return sampled_a + sampled_c*((X - sampled_b)**2) # x.shape is 


compute_y()
