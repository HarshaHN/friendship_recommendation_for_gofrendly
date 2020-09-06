

#%%--------------------
""" 01. Embedding similarity distribution """

def embplot(embs):
  # input: embs, output: a plot
  from sklearn.metrics.pairwise import cosine_similarity
  import numpy as np
  import random

  for emb in embs:
    emb = emb.numpy()
    ind = random.sample(range(len(emb)), 2000)
    iemb = emb[ind]
    s=[]; limit = len(iemb)-1
    for i,a in enumerate(iemb):
        if i == limit: break
        val = cosine_similarity([a], iemb[i+1:])[0]
        s.extend(val)
  return s

#%%--------------------














