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


winner, loser = G[0, :]

W[winner], W[loser]


pv = 0.5 * np.ones(M)   # prior variance
w = np.zeros(M)         # skills, initialized to be the prior mean μ0 = 0
skills_across_iterations = [w]

for iteration in tqdm(range(4000)):
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


with open('skills.txt','wb') as f:
    for sk in skills_across_iterations:
        np.savetxt(f, sk, fmt='get_ipython().run_line_magic(".2f')", "")


[(pl[0], np.mean(pl[1])) for pl in player_subset]


fig = plt.figure(figsize=(30,6))

num_iterations = list(range(len(skills_across_iterations)))
player_subset = [(W[i], [skills_across_iterations[j][i] for j in range(len(skills_across_iterations))]) for i in range(6)]

for i in range(6):
    ax = fig.add_subplot(2, 3, i+1) # 2 rows of panels, 3 columns
    ax.plot(num_iterations, player_subset[i][1])
    ax.set_title(f'{player_subset[i][0]}', fontsize=10)


fig = plt.figure(figsize=(30,6))

num_iterations = list(range(len(skills_across_iterations)))
player_subset = [(W[i], [skill_progression[i] for skill_progression in skills_across_iterations]) for i in range(6)]

for i in range(6):
    ax = fig.add_subplot(2, 3, i+1) # 2 rows of panels, 3 columns
    ax.acorr(player_subset[i][1] - np.mean(player_subset[i][1]), maxlags=40)
    ax.set_title(f'{player_subset[i][0]}', fontsize=10)


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


url = 'https://gitlab.developers.cam.ac.uk/djw1005/le49-probml/-/raw/master/data/coal.txt'
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
