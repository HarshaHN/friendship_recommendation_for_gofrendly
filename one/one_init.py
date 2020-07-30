
#%%-----------------------------------
"""
Date: 2 June 2020
Author: Harsha harshahn@kth.se
Friendship recommendations based on user profile representation; output: [hitrate, mrr]
"""

#%%-----------------------------------
""" 01. User features into Torch tensor """
#import dproc
import torch
import pickle
import pandas as pd
import numpy as np

# sqlusers = pd.read_hdf("../data/one/sqlusers.h5", key='01') # dproc.py: preproc >> feature
# ['user_id', 'story', 'iam', 'meetfor', 'birthday', 'marital', 'children', 'lat', 'lng']

# 01: processed user features ['story', 'iam', 'meetfor', 'age', 'marital', 'kids', 'lat','lng']
feat_df = pd.read_hdf("../data/one/user_features.h5", key='02') # ['emb', 'cat', 'num']
feat_df['emb'] = feat_df['emb'].apply(lambda x: np.zeros(768) if type(x)==int else x)

num_data = torch.tensor(list(feat_df.num), dtype=torch.float32)
cat_data = torch.tensor(list(feat_df.cat), dtype=torch.float32)
sbert_emb = torch.tensor(list(feat_df.emb), dtype=torch.float32)
X = torch.cat((num_data, cat_data, sbert_emb), 1)

#with open('../data/one/X.pkl', 'wb') as f: pickle.dump(X, f)

del cat_data, num_data, sbert_emb, f

#%%-----------------------------------
""" 02. Load the network connections """

# with open('../data/one/rawidx_nw.pkl', 'rb') as f: [trainpos, trainneg] = pickle.load(f)
# 72382, 402761: (16063,56319), (2134,400627)
# with open('../data/one/rawidx_valnw.pkl', 'rb') as f: valpos = pickle.load(f)

id_idx = {id: n for n,id in enumerate(feat.index)} # dict(enumerate(feat.index))
trainpos = [(id_idx[a], id_idx[b]) for a,b in trainpos ]
trainneg = [(id_idx[a], id_idx[b]) for a,b in trainneg ]
valpos = [(id_idx[a], id_idx[b]) for a,b in valpos]

with open('../data/one/nw.pkl', 'rb') as f: [trainpos, trainneg] = pickle.load(f) # 16063, 2134
with open('../data/one/valpos.pkl', 'rb') as f: valpos = pickle.load(f) #904

del f

#misc: with open('colab.pkl', 'rb') as f: [G, X, trainpos, trainneg] = pickle.load(f)

#%%----------------------------
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

#%%---------------------
from viz import embplot
import matplotlib.pyplot as plt
import seaborn as sns
  
#x = embplot([X[:,3:]]); plt.figure(3); sns.kdeplot(x)
#x = embplot([X[:,:3]]); plt.figure(3); sns.kdeplot(x)
#x = embplot([X]); plt.figure(3); sns.kdeplot(x)

#%%-----------------------------------
""" 03. Encoder Model """
import pipe
import nn
import torch
import matplotlib.pyplot as plt 
import importlib; importlib.reload(nn); importlib.reload(pipe)
import seaborn as sns
import time

testneg = neg 
testpos = pos
# embed = nn.Embedding(*X.shape)
onemodel = nn.net(  inputs = X, #embed.weight
                    output_size = 48, #6, 10
                    layers = [], #48,24,12  #48,36,30 
                    dropout = 0.1,
                    lr = 1e-3, #1e-2, 2e-3
                    opt = 'RMSprop', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                    cosine_lossmargin = 0.25, #-1, -0.5
                    pos = testpos, #72382: (16063,56319)
                    neg = testneg) #402761: (2134,400627)

run = 1; train=[]; val=[]; exp=[]
if run==0:
  t0 = time.time()
  emb = onemodel.train(epochs=1, lossth=0.01)
  onepipe = pipe.pipeflow(emb, K=500, nntype='cosine')
  train.append(onepipe.dfmanip(testpos)[0])
  val.append(onepipe.dfmanip(valmfs)[0])

  e = embplot([emb]); plt.figure(4); sns.kdeplot(e)
  print('Time taken:', time.time()-t0)


#%%--------------------
""" 03. a. Training """
for i in range(4):
  t1 = time.time()
  onemodel.optimizer = getattr(torch.optim, 'RMSprop')(onemodel.net.parameters(), 2e-3/(2*i+1))
  emb = onemodel.train(epochs = 50, lossth=0.01)
  onepipe = pipe.pipeflow(emb, K=500, nntype='cosine')
  train.append(onepipe.dfmanip(testpos)[0])
  #exp.append(onepipe.dfmanip(trainpos)[0])
  val.append(onepipe.dfmanip(valmfs)[0])
  e = embplot([emb]); plt.figure(4); sns.kdeplot(e)
  print('Time taken by train round', i, '=', time.time()-t1)

fig2 = plt.figure(2); plt.plot(train); plt.plot(val); plt.legend()
fig2.suptitle('Hitrate Vs. Epoch*50')
plt.xlabel('Epoch'); plt.ylabel('Hitrate');
#plt.figure(3); plt.plot(exp)
print(emb.mean(0)); print(emb.std(0))
# with open('one.pkl', 'wb') as f: pickle.dump([emb], f)


#%%=================================================
"""
Date: 6 May 2020
Author: Harsha harshahn@kth.se
Graph Neural Network, output: [hitrate, mrr]
"""

#%%-------------------------------------------------
# Load the files
import pickle
import dgl
with open('../data/colab.pkl', 'rb') as f: [_, X, _, _] = pickle.load(f) #[G, X, trainpos, trainneg]
#with open('../data/X.pkl', 'rb') as f: X = pickle.load(f)
with open('../data/nw.pkl', 'rb') as f: [pos, neg] = pickle.load(f)
with open('../data/valpos.pkl', 'rb') as f: valpos = pickle.load(f)
#with open('one.pkl', 'rb') as f: [emb] = pickle.load(f)

#with open('data/all_X.pkl', 'rb') as f: X = pickle.load(f)
#with open('data/all_nw.pkl', 'rb') as f: [pos, neg] = pickle.load(f)
#with open('data/all_valpos.pkl', 'rb') as f: valpos = pickle.load(f)

del f

#%%---------------
import torch
import os
print(torch.cuda.is_available()) #False
print(os.cpu_count()) #4
#print(torch.cuda.current_device()) #0
#print(torch.cuda.device(0)) #<torch.cuda.device at 0x7fc1f955b198>
#print(torch.cuda.device_count()) #1
#print(torch.cuda.get_device_name(0)) #'Tesla T4'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)


#%%------------------------------------
"""Move the tensors to GPU """
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)
#device = torch.device('cuda')
pos, neg, valpos = torch.tensor(pos), torch.tensor(neg), torch.tensor(valpos)
X, pos, neg, valpos = X.to(device), pos.to(device), neg.to(device), valpos.to(device)
print(X.is_cuda, pos.is_cuda, neg.is_cuda, valpos.is_cuda)

#%%----------------------------------------------
""" 01. Embedding similarity distribution """

def embplot(embs):
  # input: embs, output: sim distrib
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


#%%-------------------------
import matplotlib.pyplot as plt
import seaborn as sns

#x = embplot([X[:,3:]]); plt.figure(3); sns.kdeplot(x)
#x = embplot([X[:,:3]]); plt.figure(3); sns.kdeplot(x)
#x = embplot([X]); plt.figure(3); sns.kdeplot(x)

#%%---------------------------------------
# Make DGL graph
import dgl
def makedgl(num, pos):
    G = dgl.DGLGraph()
    G.add_nodes(num)
    G.add_edges(G.nodes(), G.nodes()) #self loop all
    G.add_edges(*zip(*pos)) #add edges list(zip(*pos))
    G = dgl.to_bidirected(G) 
    G = dgl.graph(G.edges(), 'user', 'frd')
    print('-> Graph G has %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 
    return G

G = makedgl(num=len(X), pos=pos)

#%%-------------------------------------------------
# Random walk
import dgl
import torch
rw = dgl.sampling.RandomWalkNeighborSampler(G=G, 
                                            random_walk_length=64, 
                                            random_walk_restart_prob=0,
                                            num_random_walks=8,
                                            num_neighbors=16,
                                            weight_column='w')

rwG = rw(torch.LongTensor(G.nodes()))
rwG.edata['w'] = rwG.edata['w'].float()

del rw, rwG.edata['_ID'], G 
print('-> Graph G has %d nodes' % rwG.number_of_nodes(), 'with %d edges' % (rwG.number_of_edges())) 
# ng.predecessors(1) # ng.edata['w'][ng.edge_ids(*ng.in_edges(1))]

#%%----------------------
""" 01. Graph Neural Network """
# rwG.is_readonly
import nn
import pipe
import torch
import matplotlib.pyplot as plt 
import importlib; importlib.reload(nn); importlib.reload(pipe)
import seaborn as sns
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print(device)

# fdim = X.shape[1]
neg = neg #402761: (2134,400627)
pos = pos #72382: (16063,56319)
twomodel = nn.gnet( graph = rwG,
                    nodeemb = X, #X[:,:3],
                    convlayers = [[48,48]],#, [48,45,45], [45,42,42], [42,39,39], [39,36,36]],
                    output_size = 48,
                    dropout = 0.00,
                    lr = 5e-4,
                    opt = 'RMSprop', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                    select_loss = 'cosine', #'pinsage'
                    loss_margin = 0.25, # 0.25
                    pos = pos, #16063
                    neg = neg, #2134
                    fdim = 48)

twomodel.to(device)

#trainE = []; valE = []
trainE=[28.2]; valE=[26.4]
twopipe = pipe.pipeflow(X, K=500, nntype='cosine')

#%%----------------------------------
import pipe
import importlib; importlib.reload(pipe)
import time

trainE, valE = [], []
t0 = time.time()

twopipe = pipe.pipeflow(X, K=500)
trainE.append(twopipe.dfmanip(pos)[0])
valE.append(twopipe.dfmanip(valpos)[0])

# Raw X: # Hitrate = 28.2 MRR = 2.9; Hitrate = 26.4 MRR = 1.3 (173s)
# node-based: Hitrate = 26.3 MRR = 2.4; Hitrate = 27.3 MRR = 0.8
"""
newemb = twomodel.train(epochs=1, lossth=0.01)
print('Grad / Weights =', (list(twomodel.net.parameters())[1].grad / list(twomodel.net.parameters())[1]).mean())
twopipe = pipe.pipeflow(newemb, K=500, nntype='cosine')
trainE.append(twopipe.dfmanip(pos)[0])
valE.append(twopipe.dfmanip(valpos)[0])

e = embplot([newemb]); plt.figure(4); sns.kdeplot(e)
"""
print('Time taken:', time.time()-t0)

# Raw X: # Hitrate = 28.2 MRR = 2.9; Hitrate = 26.4 MRR = 1.3
# Hitrate = 29.2 MRR = 3.4; Hitrate = 21.9 MRR = 1.3



#%%--------------------
""" 03. a. Training """
for i in range(3):
  t1 = time.time()
  twomodel.optimizer  = getattr(torch.optim, 'RMSprop')(twomodel.net.parameters(), 5e-4) #5e-4
  newemb = twomodel.train(epochs=30, lossth=0.01)
  print('Grad / Weights =', (list(twomodel.net.parameters())[1].grad / list(twomodel.net.parameters())[1]).mean())
  
  twopipe = pipe.pipeflow(newemb, K=500, nntype='cosine')
  trainE.append(twopipe.dfmanip(pos)[0])
  valE.append(twopipe.dfmanip(valpos)[0])

  fig2 = plt.figure(2); e = embplot([newemb]); sns.kdeplot(e)
  fig2.suptitle('Node similarity distribution'); plt.xlabel('cosine similarity'); #plt.ylabel('Hitrate')
  print('Time taken by train round', i, '=', time.time()-t1)

fig3 = plt.figure(3); plt.plot(trainE); plt.plot(valE); plt.legend()
fig3.suptitle('Hitrate Vs. Epoch'); plt.xlabel('Epoch'); plt.ylabel('Hitrate')
#fig3.savefig('test.jpg')

print(newemb.mean(0)); print(newemb.std(0))

#%%---------------------
with open('data/ttwo.pkl', 'wb') as f: pickle.dump(newemb, f)

#%%----------------------------------
import torch
import time
import torch.nn.functional as F

#input1 = torch.randn(5, 2); print(input1)
#input2 = torch.randn(1, 2); print(input2)
t0 = time.time()
for i in range(X.shape[0]):
  output = F.cosine_similarity(X, X[i][None,:]);#print(output)
  out = torch.topk(output, 10); #print(out.indices)

print('Time taken:', time.time()-t0)

#%%---------------
import numpy
import nearpy

#from nearpy import Engine
#from nearpy.hashes import RandomBinaryProjections

# Dimension of our vector space
dimension = 48

# Create a random binary hash with 10 bits
rbp = nearpy.hashes.RandomBinaryProjections('rbp', 10)

# Create engine with pipeline configuration
engine = nearpy.Engine(dimension, lshashes=[rbp])

# Index 1000000 random vectors (set their data to a unique string)
engine.store_many_vectors(X.numpy())
"""
for index in range(100000):
    v = numpy.random.randn(dimension)
    engine.store_vector(v, 'data_%d' % index)
"""
# Create random query vector
#query = X[2].numpy() #numpy.random.randn(dimension)

# Get nearest neighbours
#N = engine.neighbours(query)



#%%-----------
import sys
sys.modules[__name__].__dict__.clear()
