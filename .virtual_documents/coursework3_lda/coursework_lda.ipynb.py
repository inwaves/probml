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
    A,B = [pandas.DataFrame({'doc_id': M[:,0]-1, 'word_id': M[:,1]-1, 'count': M[:,2]}, 
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


gamma(0.1*vocabulary_size)


def per_word_perplexity(p_w, n):
    return -np.log2(p_w)/n

def per_Word_likelihood(p_w, n):
    return np.log(p_w)/n





def bmm_generate(doc_length, V, α, γ, K):
    # doc_length = [num words in doc1, num words in doc2, ...]
    θ = np.random.dirichlet(α * np.ones(K))              # prob dist over document classes {1,...,K}
    β = np.random.dirichlet(γ * np.ones(len(V)), size=K) # for each doc class, a prob dist over words
    z = np.random.choice(K, p=θ, size=len(doc_length))   # doc class of each document
    return [np.random.choice(V, p=β[zd], size=nd) for zd,nd in zip(z, doc_length)]

for doc in bmm_generate(doc_length=[5,2,4], V=V, α=10, γ=.1, K=20):
    print(doc)


def bmm_gibbs(doc_label, word_id, count, W, α, γ, K):
    """word_id : Series of word_ids
       count : Series of counts
       doc_label : Series of document IDs
    """
    # doc_labels = distinct values of doc_label
    # doc_index = a list as long as doc_label
    #             such that doc_labels[doc_index[j]] = doc_label[j]
    doc_labels, doc_index = np.unique(doc_label, return_inverse=True)

    # z[i] = class of document i, where i enumerates the distinct doc_labels
    # doc_count[k] = number of documents of class k
    z = np.random.choice(K, len(doc_labels))
    doc_count = np.zeros(K, dtype=int)
    for k in z: doc_count[k] += 1

    # A DataFrame indexed by document class that is used to count occurrences
    # of each word in documents of class k.
    x = pandas.DataFrame({'doc_class': z[doc_index], 'word_id': word_id, 'count': count}) \
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
            # Why remove?
            w,c = word_id[doc_index==i].values, count[doc_index==i].values
            occurrences[z[i], w] -= c
            word_count[z[i]] -= sum(c)
            doc_count[z[i]] -= 1

            # Find the log probability that this document belongs to class k, marginalized over θ and β
            logp = [... for k in range(K)]
            p = np.exp(logp - np.max(logp)) # Why do we do this? Is it to renormalise?
            p = p/sum(p)

            # Assign this document to a new class, chosen randomly, and add back the counts
            k = np.random.choice(K, p=p)
            z[i] = k
            occurrences[k, w] += c
            word_count[k] += sum(c)
            doc_count[k] += 1
        
        yield np.copy(z)


g = bmm_gibbs(A['doc_id'], A['word_id'], A['count'], W=len(V), α=10, γ=.1, K=20)
NUM_ITERATIONS = 20
res = np.stack([next(g) for _ in range(NUM_ITERATIONS)])
# this produces a matrix with one row per iteration and a column for each unique doc_id
