import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import scipy.io, scipy.stats, scipy.linalg
from numpy.linalg import solve
import requests, io
from tqdm import tqdm


r = requests.get('https://www.cl.cam.ac.uk/teaching/2122/DataSci/data/tennis_data.mat')
with io.BytesIO(r.content) as f:
    data = scipy.io.loadmat(f)
    W = np.concatenate(data['W'].squeeze())
    G = data['G'] - 1   # W[G[i,0]] is winner of game i, W[G[i,1]] is loser
    M = W.shape[0]      # number of players M = 107
    N = G.shape[0]      # number of games N = 1801


pv = 0.75 * np.ones(M)   # prior variance
w = np.zeros(M)         # skills, initialized to be the prior mean μ0 = 0
skills_across_iterations = [w]

for iteration in tqdm(range(1100)):
    # Sample performance differences (t) given skills (w) and outcomes (G)
    s = w[G[:,0]] - w[G[:,1]] # Deterministic skill difference.
    σ = 1
    
    # Skill difference plus some noise epsilon.
    t = s + σ * scipy.stats.norm.ppf(1 - np.random.uniform(size=N)*(1-scipy.stats.norm.cdf(-s/σ)))

    # Sample skills given performance differences, i.e. 
    # Find some covariance and mean and distribute the new skills according to that.
    inverse_Sigma_tilde = np.zeros((M, M))
    mu_tilde = np.zeros(M)
    
    # This would compute each game precision matrix and add it to the inverse Σ~
    # But here it just happens directly for conciseness.
    for g in range(N):
        inverse_Sigma_tilde[G[g, 0], G[g, 0]] += 1
        inverse_Sigma_tilde[G[g, 1], G[g, 1]] += 1
        inverse_Sigma_tilde[G[g, 0], G[g, 1]] -= 1
        inverse_Sigma_tilde[G[g, 1], G[g, 0]] -= 1
        
    # Compute μ~: for all players i in M, for all games g in G, if the player won game g, add t[g], otherwise add -t[g] to mu_tilde[i]
    for i in range(M):
        for g in range(N):
            if G[g, 0] == i:
                mu_tilde[i] += t[g]
            elif G[g, 1] == i:
                mu_tilde[i] -= t[g]
        
    # Saving a call to np.linalg.inv by just inverting the diagonal covariance Sigma_0 directly.
    Σinv = np.diag(1/pv) + inverse_Sigma_tilde
    Σ = np.linalg.inv(Σinv)
    μ = Σ @ mu_tilde
    w = np.random.multivariate_normal(mean=μ, cov=Σ)
    
    # Storing results across iterations for plotting.
    skills_across_iterations.append(w)



