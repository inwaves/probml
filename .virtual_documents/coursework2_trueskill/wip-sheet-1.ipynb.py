import numpy as np
import pandas
import matplotlib
import matplotlib.pyplot as plt
import scipy.io, scipy.stats, scipy.linalg
from numpy.linalg import solve
from tqdm import tqdm

import requests, io


r = requests.get('https://www.cl.cam.ac.uk/teaching/2122/DataSci/data/tennis_data.mat')
with io.BytesIO(r.content) as f:
    data = scipy.io.loadmat(f)
    W = np.concatenate(data['W'].squeeze())
    G = data['G'] - 1   # W[G[i,0]] is winner of game i, W[G[i,1]] is loser
    M = W.shape[0]      # number of players M = 107
    N = G.shape[0]      # number of games N = 1801


pv = 0.5 * np.ones(M)   # prior variance
w = np.zeros(M)         # skills, initialized to be the prior mean μ0 = 0
skills_across_iterations = [w]

for iteration in tqdm(range(1100)):
    # Sample performance differences (t) given skills (w) and outcomes (G)
    s = w[G[:,0]] - w[G[:,1]] # s is the skill difference.
    σ = 1
    
    # Sample t according to the skill difference plus some noise epsilon.
    t = s + σ * scipy.stats.norm.ppf(1 - np.random.uniform(size=N)*(1-scipy.stats.norm.cdf(-s/σ)))

    # Sample skills given performance differences, i.e. 
    # Find some mean and covariance and distribute the new skills according to that.
    inverse_Sigma_tilde = np.zeros((M, M))
    mu_tilde = np.zeros(M)
    
    # Compute Σ~^{-1}
    for g in range(N):
        inverse_Sigma_tilde[G[g, 0], G[g, 0]] += 1
        inverse_Sigma_tilde[G[g, 1], G[g, 1]] += 1
        inverse_Sigma_tilde[G[g, 0], G[g, 1]] -= 1
        inverse_Sigma_tilde[G[g, 1], G[g, 0]] -= 1
        
    # Compute μ~
    for i in range(M):
        for g in range(N):
            if G[g, 0] == i:
                mu_tilde[i] += t[g]
            elif G[g, 1] == i:
                mu_tilde[i] -= t[g]
        
    Σinv = np.diag(1/pv) + inverse_Sigma_tilde
    Σ = np.linalg.inv(Σinv)
    μ = Σ @ mu_tilde
    w = np.random.multivariate_normal(mean=μ, cov=Σ)
    
    # Storing results across iterations for plotting.
    skills_across_iterations.append(w)


## This cell is used by multiple experiments below, it's likely you will need to run it.
num_iterations = list(range(len(skills_across_iterations)))

# Selecting a subset or players whose skill samples to look at.
players = [0, 1, 2, 3, 4, 15]
player_subset = [(W[i], [skills_across_iterations[j][i] for j in range(len(skills_across_iterations))]) for i in players]


fig = plt.figure(figsize=(30,6))
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1) # 2 rows of panels, 3 columns
    ax.plot(num_iterations, player_subset[i][1])
    ax.set_title(f'{player_subset[i][0]}', fontsize=10)


fig = plt.figure(figsize=(30,6))

for i in range(6):
    ax = fig.add_subplot(2, 3, i+1) # 2 rows of panels, 3 columns
    ax.acorr(player_subset[i][1] - np.mean(player_subset[i][1]), maxlags=40)
    ax.set_title(f'{player_subset[i][0]}', fontsize=10)


# Here I try thinning by taking every 10 samples.
stride = 10
thinned_num_iterations = list(range(0, len(skills_across_iterations), stride))
thinned_player_subset = [(W[i], [skills_across_iterations[j][i] for j in range(0, len(skills_across_iterations), stride)]) for i in players]

fig = plt.figure(figsize=(30,6))
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1) # 2 rows of panels, 3 columns
    ax.plot(thinned_num_iterations, thinned_player_subset[i][1])
    ax.set_title(f'{thinned_player_subset[i][0]}', fontsize=10)


def gaussian_ep(G, M):
    def Ψ(x): return scipy.stats.norm.pdf(x) / scipy.stats.norm.cdf(x)
    def Λ(x): return Ψ(x) * (Ψ(x) + x)
    N = len(G)

    μ_s, p_s = np.empty(M), np.empty(M)
    μ_gs, p_gs = np.zeros((N,2)), np.zeros((N,2))
    μ_sg, p_sg = np.empty((N,2)), np.empty((N,2))
    
    while True:
        # 1. Compute marginal skills
        # Let skills be N(μ_s, 1/p_s)
        p_s = np.ones(M) * 1/0.5
        μ_s = np.zeros(M)
        for j,(winner,loser) in enumerate(G):
            p_s[winner] += p_gs[j,0]
            p_s[loser] += p_gs[j,1]
            μ_s[winner] += μ_gs[j,0] * p_gs[j,0]
            μ_s[loser] += μ_gs[j,1] * p_gs[j,1]
        μ_s = μ_s / p_s

        # 2. Compute skill -> game messages
        # winner's skill -> game: N(μ_sg[,0], 1/p_sg[,0])
        # loser's skill -> game: N(μ_sg[,1], 1/p_sg[,1])
        p_sg = p_s[G] - p_gs
        μ_sg = (p_s[G]*μ_s[G] - p_gs*μ_gs) / p_sg

        # 3. Compute game -> performance messages
        v_gt = 1 + np.sum(1/p_sg, 1)
        σ_gt = np.sqrt(v_gt)
        μ_gt = μ_sg[:,0] - μ_sg[:,1]

        # 4. Approximate the marginal on performance differences
        μ_t = μ_gt + σ_gt * Ψ(μ_gt/σ_gt)
        p_t = 1 / v_gt / (1-Λ(μ_gt/σ_gt))

        # 5. Compute performance -> game messages
        p_tg = p_t - 1/v_gt
        μ_tg = (μ_t*p_t - μ_gt/v_gt) / p_tg

        # 6. Compute game -> skills messages
        # game -> winner's skill: N(μ_gs[,0], 1/p_gs[,0])
        # game -> loser's skill: N(μ_gs[,1], 1/p_gs[,1])
        p_gs[:,0] = 1 / (1 + 1/p_tg + 1/p_sg[:,1])  # winners
        p_gs[:,1] = 1 / (1 + 1/p_tg + 1/p_sg[:,0])  # losers
        μ_gs[:,0] = μ_sg[:,1] + μ_tg
        μ_gs[:,1] = μ_sg[:,0] - μ_tg
        
        yield (μ_s, np.sqrt(1/p_s))


g = gaussian_ep(G, M)
mu_across_iterations, sigma_across_iterations = [], []

for it in range(num_iter):
    mu_s, sig_s = next(g)
    mu_across_iterations.append(mu_s)
    sigma_across_iterations.append(sig_s)

# For each of our selected players, get their μ, σ across the iterations of message passing.
player_mu_sig = [(W[i], [mu_across_iterations[j][i] for j in range(len(mu_across_iterations))], [sigma_across_iterations[j][i] for j in range(len(sigma_across_iterations))]) for i in players]

fig = plt.figure(figsize=(30,6))
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1) # 2 rows of panels, 3 columns
    ax.plot(iterations, player_mu_sig[i][1], label="μ")
    ax.plot(iterations, player_mu_sig[i][2], label="σ")
    ax.legend()
    ax.set_title(f'{player_mu_sig[i][0]}', fontsize=10)


player_at_convergence = [] # List of player name, their skill mean, and their skill standard deviation.
for player in player_mu_sig:
    player_at_convergence.append((player[0], player[1][-1], player[2][-1]))
    
fig = plt.figure(figsize=(30,8))

for i in range(6):
    mu, sigma = player_at_convergence[i][1], player_at_convergence[i][2]
    ax = fig.add_subplot(2, 3, i+1)
    ax.set_title(player_subset[i][0])
    ax.plot(num_iterations[1:], player_subset[i][1][1:], alpha=0.5, label="Gibbs samples")
    ax.plot(num_iterations[1:], [mu]*1100, label="μ")
    ax.fill_between(num_iterations[1:], mu-2*sigma, mu+2*sigma, color='yellow', alpha=0.3, label="2σ")
    ax.legend()


nadal_marginal_samples = np.array(player_subset[0][1][10:])
djokovic_marginal_samples = np.array(player_subset[-1][1][10:])


plt.hist(djokovic_marginal_samples, label="Djokovic")
plt.hist(nadal_marginal_samples, label="Nadal")
plt.legend()
plt.show()


mu_n = nadal_marginal_samples.mean()
sd_n = 0.2 # Eyeballing the histogram of (marginal samples - their mean) shows σ_n approximately equal to 0.2.

nadal_recovered_dist = scipy.stats.norm(mu_n, sd_n)
x_n = np.linspace(mu_n - 3 * sd_n, mu_n + 3 * sd_n)

# Now do the same for Djokovic.
mu_d = djokovic_marginal_samples.mean()
sd_d = 0.22 # Eyeballing the histogram of (marginal samples - their mean) shows σ_s approximately equal to 0.22.

djokovic_recovered_dist = scipy.stats.norm(mu_d, sd_d)
x_d = np.linspace(mu_d - 3 * sd_d, mu_d + 3 * sd_d)


mu_diff = mu_n - mu_d
sd_diff = np.sqrt(sd_n**2 + sd_d**2)
W_d = scipy.stats.norm(mu_diff, sd_diff)

# Probability that w_d is positive.
f"p(W0 > W15) = {W_d.sf(0):.3f}, using reconstructed marginals."


# To find the probability directly from the joint distribution, we can count in how many of the samples w0 > w15.
nadal_higher_skill = 0
for sample in skills_across_iterations[10:]:
    if sample[0] > sample[15]:
        nadal_higher_skill += 1

f"p(W0 > W15) = {nadal_higher_skill/(len(skills_across_iterations)-10):.3f}, corresponding to {nadal_higher_skill}/{len(skills_across_iterations)-10} samples."


plt.scatter(nadal_marginal, djokovic_marginal, alpha=0.6, color='teal')
plt.xlabel("nadal_skill")
plt.ylabel("djokovic_skill")
plt.plot([1, 2], [1, 2], color='orange')


mu_n, sigma_n = player_at_convergence[0][1], player_at_convergence[0][2]
mu_d, sigma_d = player_at_convergence[-1][1], player_at_convergence[-1][2]

x1 = np.linspace(mu_n - 3*sigma_n, mu_n + 3*sigma_n)
x2 = np.linspace(mu_d - 3*sigma_d, mu_d + 3*sigma_d)
plt.fill(x1, scipy.stats.norm.pdf(x1, mu_n, sigma_n), alpha=0.7, label="Nadal skill dist.")
plt.fill(x2, scipy.stats.norm.pdf(x2, mu_d, sigma_d), alpha=0.7, label="Djokovic skill dist.")
plt.legend()
print(f"μ_n={mu_n**2} μ_d={mu_d**2}")


# We want to find out the probability p(W0 - W15) > 0, where
# W0 ~ N(µ_n, σ_n**2), W15 ~ N(µ_d, σ_d**2).
# Wd = W0 - W15 is ~ N(mu_d - mu_n, sqrt(σ_d**2 + σ_n **2)).
mu_difference = mu_n - mu_d
sd_difference = np.sqrt(sigma_n**2 + sigma_d **2)

# The probability that Wd > 0 is 1 - Φ(0), also
# known as the survival function.
f"p(w0 > w15) = {scipy.stats.norm(mu_difference, sd_difference).sf(0):.3f}"


# We have some samples for w0, some for w15. We get t ~ N(w0-w15, 1)
# Nadal beats Djokovic iff t > 0, so 1-Φ(0). As below, but samples are Gibbs.
mu_t0 = np.mean(nadal_marginal - djokovic_marginal)
t0 = scipy.stats.norm(mu_t0, 1)

x_t0 = np.linspace(mu_t0 - 3, mu_t0 + 3)
plt.fill(x_t0, t0.pdf(x_t0), alpha=0.7)


# 1 - Φ_t0(0)
f"p(t > 0 | w0, w15) = {t0.sf(0):.3f}" 


cumulative_probability = 0
for sample in skills_across_iterations:
    t = scipy.stats.norm(sample[0]-sample[15], 1)
    cumulative_probability += t.sf(0)
    
f"p(t > 0 | w0, w15) = {cumulative_probability / len(skills_across_iterations):.3f}"


# Get some samples for w0 and w15 from the distributions output by message passing.
nadal_skill = scipy.stats.norm(mu_n, sigma_n)
djokovic_skill = scipy.stats.norm(mu_d, sigma_d)

mu_t1 = np.mean(nadal_skill.rvs(size=10000) - djokovic_skill.rvs(size=10000))
t1 = scipy.stats.norm(mu_t1, 1)

x_t1 = np.linspace(mu_t1 - 3, mu_t1 + 3)
plt.fill(x_t1, t1.pdf(x_t1), alpha=0.7)

plt.plot([0, 0], [0, 0.4], color='orange', alpha=0.5)


# 1 - Φ_t(0)
f"p(t > 0 | w0, w15) = {t1.sf(0):.3f}" 


url = 'https://www.cl.cam.ac.uk/teaching/2122/DataSci/data/coal.txt'
coal = pandas.read_csv(url, names=['date'], parse_dates=['date'])

# Count number of explosions by year, and fill in missing years with count=0
counts = coal['date'].groupby(coal['date'].dt.year).apply(len)
y,Y = np.min(counts.index.values), np.max(counts.index.values)
df = pandas.DataFrame({'count': np.zeros(Y-y+1, dtype=int)}, index=np.arange(y,Y+1))
df.index.name = 'year'
df.loc[counts.index, 'count'] = counts
# Discard first and last year (for which counts might be incomplete)
df = df.reset_index()
df = df.loc[(df['year']>y) & (df['year']<Y)]

with matplotlib.rc_context({'figure.figsize': [12,2]}):
    plt.bar(df['year'].values, df['count'].values)
    plt.title('mine explosions per year')
plt.show()


l = scipy.stats.gamma


wins = np.zeros(M, dtype=int)
losses = np.zeros(M, dtype=int)
w, x = np.unique(G[:,0], return_counts=True)
wins[w] = x
w, x = np.unique(G[:,1], return_counts=True)
losses[w] = x
score = wins / (wins + losses)
rank_order = np.argsort(score)[::-1]

fig,ax = plt.subplots(figsize=(20,5))
x = np.arange(M)
ax.bar(x, score[rank_order], align='center', width=.8)
ax.set_xticks(x)
ax.set_xticklabels(W[rank_order], rotation=-90, ha='right', fontsize=8)
ax.set_ylabel('get_ipython().run_line_magic("win')", "")
plt.show()


fig = plt.figure(figsize=(6,4))
for i in range(5):
    ax = fig.add_subplot(2, 3, i+1) # 2 rows of panels, 3 columns
    ax.plot(np.random.normal(size=10), np.random.normal(size=10))
    ax.set_title(f'Panel {i}', fontsize=10)


# Generate synthetic data with autocorrelations
x = [5, 5]
for _ in range(1000):
    lastx = x[-2:]
    nextx = np.mean(lastx) + 0.1 * np.random.normal() - 0.3 * (lastx[0] - 3)
    x.append(nextx)
x = np.array(x)
    
# Subtract the mean, then plot autocorrelation
plt.acorr(x - np.mean(x), maxlags=40)
plt.xlim(-1,40)
plt.show()
