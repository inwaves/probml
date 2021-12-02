import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import requests, io
from scipy.special import gamma

pd.set_option("display.max_rows", 200, "display.max_columns", 200)


r = requests.get('https://www.cl.cam.ac.uk/teaching/2122/DataSci/data/kos_doc_data.mat')
with io.BytesIO(r.content) as f:
    data = scipy.io.loadmat(f)
    V = np.array([i[0] for i in data['V'].squeeze()])
    A,B = [pd.DataFrame({'doc_id': M[:,0]-1, 'word_id': M[:,1]-1, 'count': M[:,2]}, 
                            columns=['doc_id','word_id','count']) 
           for M in (data['A'],data['B'])]


vocabulary_size = V.shape[0]
wordcount_entire_collection = A["count"].sum()
beta = np.zeros((vocabulary_size,))

for m in range(vocabulary_size):
    c_m = A[(A.word_id == m)==True]["count"].sum()
    beta[m] = c_m / wordcount_entire_collection


beta_dictionary = dict(zip(range(vocabulary_size), beta))
beta_indices_sorted_ascending = sorted(beta_dictionary, key=lambda k: beta_dictionary[k])

beta[beta_indices_sorted_ascending[-20:]]


fig, ax = plt.subplots(figsize=(5,8))
ax.barh(np.arange(20), beta[beta_indices_sorted_ascending[-20:]], align='center')
ax.set_yticks(np.arange(20))
ax.set_yticklabels(V[beta_indices_sorted_ascending[-20:]])
ax.set_xlabel(r'$\hat{\beta}$')
ax.invert_yaxis()
plt.show()


print(f"""The probability of word: {V[5479]} as estimated using MLE is: {beta[5479]}"
This word is not in any training documents:
{A[A["word_id"]==5479]}
This word is in our test document: 
{B[B.doc_id==2527].loc[54865]}""")


def per_word_perplexity(p_w, n):
    # Make sure we don't try to take log 0.
    return -np.log2(p_w)/n if p_w > 0 else 0

def per_word_likelihood(p_w, n):
    return np.log(p_w)/n if p_w > 0 else 0


alpha = 0.1

# Implement the posterior predictive we got in (b).
betas = np.zeros(vocabulary_size)
for i in range(vocabulary_size):
    betas[i] = (A[A.word_id == i]["count"].sum() + alpha) / (vocabulary_size*alpha + wordcount_entire_collection)


unique_test_docs = np.unique(B.doc_id.values)
document_likelihoods = {}

# For each document d in the test set,
for doc in unique_test_docs:
    Pr = 1
    doc_df = B[B.doc_id==doc]
    
    # For each word in document d, multiply the β_v with the other words β_w
    for word in doc_df.word_id.values:
        Pr *= betas[word] ** doc_df[doc_df.word_id==word]["count"].values[0]
        
    # Store the likelihood.
    document_likelihoods[doc] = Pr


c_2000 = B[B.doc_id==2000]["count"].sum()
c_2001 = B[B.doc_id==2001]["count"].sum()
print(f"For doc_id=2000, per-word log likelihood is: {per_word_likelihoods[2000]}. It contains: {c_2000} words.")
print(f"For doc_id=2001, per-word log likelihood is: {per_word_likelihoods[2001]}. It contains: {c_2001} words.")


# Compute per-word likelihoods and perplexities.
per_word_likelihoods = dict([(i, per_word_likelihood(document_likelihoods[i], B[B.doc_id==i]["count"].sum())) for i in document_likelihoods.keys()])
per_word_perplexities = dict([(i, per_word_perplexity(document_likelihoods[i], B[B.doc_id==i]["count"].sum())) for i in document_likelihoods.keys()])


# Plot histogram of test document per-word log-Pr.
fig, ax = plt.subplots(figsize=(10,6))

# Distinct number of documents in B
ax.bar(np.arange(len(per_word_likelihoods)), per_word_likelihoods.values(), align='center')
ax.set_xticks(np.arange(len(per_word_likelihoods)))
ax.set_xlabel("doc_id")
ax.set_ylabel(r'$\log Pr(w_d)$')
plt.show()


count_likelihoods = {}
for (doc_id, logPr) in per_word_likelihoods.items():
    cts = B[B.doc_id==doc_id]["count"].sum()
    if cts not in count_likelihoods:
        count_likelihoods[cts] = logPr
    else:
        count_likelihoods[cts] += logPr


fig, ax = plt.subplots(figsize=(8,4))
ax.scatter(count_likelihoods.keys(), count_likelihoods.values())
ax.set_xlabel("document length")
ax.set_ylabel(r'$\log Pr(w_d)$')
plt.show()


sorted_dict = {k: v for k, v in sorted(per_word_perplexities.items(), key=lambda item: item[1])}


B[B.doc_id==2618]["count"].sum()


V[B[B.doc_id==2618]["word_id"]]


V[B[B.doc_id==2000]["word_id"]]


def bmm_generate(doc_length, V, α, γ, K):
    # doc_length = [num words in doc1, num words in doc2, ...]
    θ = np.random.dirichlet(α * np.ones(K))              # prob dist over document classes {1,...,K}
    β = np.random.dirichlet(γ * np.ones(len(V)), size=K) # for each doc class, a prob dist over words
    z = np.random.choice(K, p=θ, size=len(doc_length))   # doc class of each document
    return [np.random.choice(V, p=β[zd], size=nd) for zd,nd in zip(z, doc_length)]

for doc in bmm_generate(doc_length=[5,2,4], V=V, α=10, γ=.1, K=20):
    print(doc)


def bmm_gibbs(doc_label, word_id, count, W, alpha, gamma, K):
    """word_id : Series of word_ids
       count : Series of counts
       doc_label : Series of document IDs
    """
    # doc_labels = distinct values of doc_label
    # doc_index = a list as long as doc_label
    #             such that doc_labels[doc_index[j]] = doc_label[j]
    doc_labels, doc_index = np.unique(doc_label, return_inverse=True)

    # This is the initial sampling for z_d.
    # z[i] = class of document i, where i enumerates the distinct doc_labels
    # doc_count[k] = number of documents of class k
    z = np.random.choice(K, len(doc_labels))
    doc_count = np.zeros(K, dtype=int)
    for k in z: doc_count[k] += 1

    # A DataFrame indexed by document class that is used to count occurrences
    # of each word in documents of class k.
    x = pd.DataFrame({'doc_class': z[doc_index], 'word_id': word_id, 'count': count}) \
        .groupby(['doc_class', 'word_id']) \
        ['count'].apply(sum) \
        .unstack(fill_value=0)
    
    # occurrences[k,w] = number of occurrences of word_id w in documents of class k
    occurrences = np.zeros((K, len(V)))
    occurrences[x.index.values.reshape((-1,1)), x.columns.values] = x
    
    # word_count[k] = total number of words in documents of class k
    word_count = np.sum(occurrences, axis=1)
    
    while True:
        for i in range(len(doc_labels)):

            # Get the words, counts for document i
            # and remove this document from the counts.
            w,c = word_id[doc_index==i].values, count[doc_index==i].values
            occurrences[z[i], w] -= c
            word_count[z[i]] -= sum(c)
            doc_count[z[i]] -= 1

            # Find the log probability that this document belongs to class k, marginalized over θ and β
#             beta_marginalisation = (gamma + word_count[k]) / (vocabulary_size*gamma + some_sum)
#             theta_marginalisation = (alpha + doc_count[k]) / (K*alpha + np.sum(doc_count))
            logp = [np.log(((gamma + np.sum(occurrences[k, w])) / (vocabulary_size*gamma + np.sum(occurrences[k]))) * ((alpha + doc_count[k]) / (K*alpha + np.sum(doc_count))))  for k in range(K)]
            p = np.exp(logp - np.max(logp))
            p = p/sum(p)

            # Assign this document to a new class, chosen randomly, and add back the counts
            k = np.random.choice(K, p=p)
            z[i] = k
            occurrences[k, w] += c
            word_count[k] += sum(c)
            doc_count[k] += 1
        
        yield np.copy(z), p


g = bmm_gibbs(A['doc_id'], A['word_id'], A['count'], W=len(V), alpha=10, gamma=.1, K=20)
NUM_ITERATIONS = 10
res = np.stack([next(g)[0] for _ in range(NUM_ITERATIONS)])
# this produces a matrix with one row per iteration and a column for each unique doc_id
