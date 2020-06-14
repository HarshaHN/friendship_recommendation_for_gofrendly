
#%%-----------------------------------
"""
Date: 2 June 2020
Author: Harsha harshahn@kth.se
Friendship recommendations based on user profile representation; output: [hitrate, mrr]
"""

#%%-----------------------------------
""" 01. Load the feature data """
import dproc
import pandas as pd

# sqlusers = pd.read_hdf("../data/one/sqlusers.h5", key='01') #['user_id', 'story', 'iam', 'meetfor', 'birthday', 'marital', 'children', 'lat', 'lng']
feat = pd.read_hdf("../data/one/trainfeat.h5", key='04') # ['emb', 'cat', 'num'] #dproc: preproc >> feature
#[trainpos, trainneg] = dproc.dproc.loadlinks() #72382, 402761: (16063,56319), (2134,400627)
[trainpos, trainneg, valmfs] = dproc.dproc.getlinks(feat.index)


#%%------------------------------
import pickle
with open('colab.pkl', 'rb') as f: [G, X, trainpos, trainneg] = pickle.load(f)
with open('mfbf.pkl', 'rb') as f: [pos, neg] = pickle.load(f)
with open('valmfs.pkl', 'rb') as f: valmfs = pickle.load(f)

#%%---------------------
from viz import embplot
import matplotlib.pyplot as plt
import seaborn as sns
  
#x = embplot([X[:,3:]]); plt.figure(3); sns.kdeplot(x)
#x = embplot([X[:,:3]]); plt.figure(3); sns.kdeplot(x)
x = embplot([X]); plt.figure(3); sns.kdeplot(x)

#%%-----------------------------------
""" 02. Transform to model inputs """
import torch

categorical_data = torch.tensor(list(feat.cat), dtype=torch.float32)
numerical_data = torch.tensor(list(feat.num), dtype=torch.float32)
X = torch.cat((numerical_data, categorical_data), 1)

ids = list(feat.index)
id_idx = {id: n for n,id in enumerate(feat.index)} # dict(enumerate(feat.index))
# trainpos = [(id_idx[a], id_idx[b]) for a,b in trainpos ]
# trainneg = [(id_idx[a], id_idx[b]) for a,b in trainneg ]
# pos = [(id_idx[a], id_idx[b]) for a,b in pos ]; #neg = [(id_idx[a], id_idx[b]) for a,b in neg ]

valmfs = [(id_idx[a], id_idx[b]) for a,b in valmfs]
# valpos = [(id_idx[a], id_idx[b]) for a,b in valpos]


del id_idx, numerical_data, categorical_data, ids, feat

#%%-----------------------------------
""" 03. Encoder Model """
import pipe
import nn
import torch
import matplotlib.pyplot as plt 
import importlib; importlib.reload(nn); importlib.reload(pipe)
import seaborn as sns

testpos = pos
testneg = neg
# embed = nn.Embedding(*X.shape).weight.detach()
onemodel = nn.net(  inputs = X,
                    output_size = 48, #6, 10
                    layers = [], #48,24,12  #48,36,30 
                    dropout = 0.1,
                    lr = 1e-3, #1e-2, 2e-3
                    opt = 'RMSprop', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                    cosine_lossmargin = 0, #-1, -0.5
                    pos = testpos, #72382: (16063,56319)
                    neg = testneg) #402761: (2134,400627)

run = 1; train=[]; val=[]
if run==0:
  emb = onemodel.train(epochs = 1, lossth=0.05)
  onepipe = pipe.pipeflow(emb, K=500, nntype='cosine')
  train.append(onepipe.dfmanip(testpos)[0])
  val.append(onepipe.dfmanip(valmfs)[0])

  #e = embplot([emb]); plt.figure(4); sns.kdeplot(e)

#%%--------------------
""" 03. a. Training """

#%%--------------------
""" 03. a. Training """
for i in range(1):
  onemodel.optimizer = getattr(torch.optim, 'RMSprop')(onemodel.net.parameters(), 2e-3/(2*i+1))
  emb = onemodel.train(epochs = 50, lossth=0.01)
  onepipe = pipe.pipeflow(emb, K=500, nntype='cosine')
  train.append(onepipe.dfmanip(testpos)[0])
  val.append(onepipe.dfmanip(valmfs)[0])
  e = embplot([emb]); plt.figure(4); sns.kdeplot(e)

plt.figure(2); plt.plot(train)
plt.figure(3); plt.plot(val)
print(emb.mean(0))
print(emb.std(0))












#%%---------------------
""" 04. Recommendations and evaluation """
from pipe import pipeflow

onepipe = pipeflow(X, K=500, nntype='cosine')
# df, hr, mrr
_ = onepipe.dfmanip(trainpos)

# import pickle; with open('../data/one/oneemb.pkl', 'wb') as f: pickle.dump([emb], f)

#%%=================================================
"""
Date: 6 May 2020
Author: Harsha harshahn@kth.se
Graph Neural Network, output: [hitrate, mrr]
"""

#%%---------------------
""" 01. Load the graph """
import dgl
import dproc

G = dproc.dproc.makedgl(num=len(X), pos=trainpos)
# G.readonly(True)

#%%----------------------------
# Need G, X, trainpos, trainneg
import pickle
with open('colab.pkl', 'wb') as f: pickle.dump([G, X, trainpos, trainneg], f)
#with open('colab.pkl', 'rb') as f: [G, X, trainpos, trainneg] = pickle.load(f)
with open('mfbf.pkl', 'wb') as f: pickle.dump([pos, neg], f)
#with open('mfbf.pkl', 'rb') as f: [pos, neg] = pickle.load(f)
with open('valmfs.pkl', 'wb') as f: pickle.dump(valmfs, f)
#with open('valmfs.pkl', 'rb') as f: valmfs = pickle.load(f)

#%%-------------------------------------------------
# Random walk
rw = dgl.sampling.RandomWalkNeighborSampler(G=G, 
                                            random_walk_length=32, 
                                            random_walk_restart_prob=0,
                                            num_random_walks=10,
                                            num_neighbors=10,
                                            weight_column='w')
rwG = rw(torch.LongTensor(G.nodes()))
rwG.edata['w'] = rwG.edata['w'].float()

del rw, rwG.edata['_ID'], G 
# ng.predecessors(1) # ng.edata['w'][ng.edge_ids(*ng.in_edges(1))]

#%%----------------------
""" 01. Graph Neural Network """
# rwG.is_readonly
import nn
from importlib import reload; reload(nn)

# fdim = X.shape[1]
twomodel = nn.gnet( graph = rwG,
                    nodeemb = X,
                    convlayers = [[48, 48], [48, 45, 42], [42, 39, 36]],
                    output_size = 36,
                    dropout = 0.1,
                    lr = 1e-4,
                    opt = 'RMSprop', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                    select_loss = 'pinsage',
                    loss_margin = 1, # -0.5
                    pos = trainpos, #72382
                    neg = trainneg,  #402761
                    fdim = 48)
#print(twomodel)

#%%--------------------
""" 03. a. Training """
twomodel.optimizer  = getattr(torch.optim, 'RMSprop')(twomodel.net.parameters(), 5e-5)
newemb = twomodel.train(epochs=50, lossth=0.05)
# list(twomodel.net.parameters())[-1] 
print(newemb[0])

#%%---------------------
""" 04. Recommendations and evaluation """

import pipe
twopipe = pipe.pipeflow(nodeemb, K=500, nntype='cosine')

# df, hr, mrr
_ = twopipe.dfmanip(trainpos)

#%%---------------------------
""" 05. Embedding similarity distribution """
# input: embs, output: a plot
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import seaborn as sns
import torch

def embplot(emb):
    #for emb in embs:
    s=[]; limit = len(emb)-1
    for i,a in enumerate(emb):
        if i == limit: break
        val = cosine_similarity([a], emb[i+1:])[0]
        s.extend(val)
        sns.kdeplot(s)

A = torch.tensor([[0, 1, 0, 0, 1], [0, 0, 1, 1, 1],[1, 1, 0, 1, 0]])
embplot(A.numpy())

#%%---------------------------
# Save variables
"""
import pickle
with open('../data/one/oneoutemb.pkl', 'wb') as f:
    pickle.dump([emb], f)
"""
#%%-----------
import sys
sys.modules[__name__].__dict__.clear()
