"""
File: main.py
Date: 22 April 2020
Author: Harsha HN harshahn@kth.se

Developed using PyTorch and DGL libraries
"""
#%% When run on Google Colab only
"""
  import os
  os.chdir('/content/drive/My Drive/gofrendly') #os.getcwd()

  !pip install dgl

  device = torch.device('cuda'); #print(device) #cuda
  print(torch.cuda.get_device_name(0)) #'Tesla P100'
  #print(torch.cuda.current_device()) #0
  #print(torch.cuda.device(0)) #<torch.cuda.device at 0x7fc1f955b198>
  #print(torch.cuda.device_count()) #1
  #print(torch.cuda.is_available()) #True
  #print(os.cpu_count()) #2
  #!nvidia-smi
"""

#%% Import libraries
import dgl
import time
import torch
import pickle

#%% Load the tensors
path = '../01_data_prep/05_data_out/'

user_features = torch.load(path + 'user_tensor.pt') # 14,948 x 816

with open(path + 'train_links.pkl', 'rb') as f: [trainpos, trainneg] = pickle.load(f) #16063, 2134
valpos = torch.load(path + 'valpos.pt') #904
testpos = torch.load(path + 'testpos.pt') #2596

#%% Move tensors to GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
user_features = user_features.to(device); print(user_features.is_cuda)

#%% Create social network graph
def makedgl(num, pos):
    G = dgl.DGLGraph()
    G.add_nodes(num)
    G.add_edges(G.nodes(), G.nodes()) #self loop all
    G.add_edges(*zip(*pos)) #add edges list(zip(*pos))
    G = dgl.to_bidirected(G) 
    G = dgl.graph(G.edges(), 'user', 'frd')
    print('-> Graph G has %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 
    return G

G = makedgl(num=len(user_features), pos=trainpos)

#%% Produce computational graph using random-walk
t0 = time.time()
rw = dgl.sampling.RandomWalkNeighborSampler(G = G, 
                                            random_walk_length = 32,
                                            random_walk_restart_prob=0,
                                            num_random_walks = 32,
                                            num_neighbors = 16,
                                            weight_column = 'w')

rwG = rw(torch.LongTensor(G.nodes()))
rwG.edata['w'] = rwG.edata['w'].float(); 

del rw, rwG.edata['_ID'], G 
print('-> Graph G has %d nodes' % rwG.number_of_nodes(), 'with %d edges' % (rwG.number_of_edges())) # rwG.is_readonly
print('Time taken:', time.time() - t0)

#%% Define GNN

import nn
import eval
 
fdim = user_features.shape[1] 
nn_model = nn.gnet( graph = rwG,
                    nodeemb = user_features,
                    convlayers = [[fdim, fdim]], #[[fdim, fdim], [fdim, fdim, fdim], [fdim, fdim, fdim]],
                    output_size = fdim,
                    dropout = 0.00,
                    lr = 3e-4,
                    opt = 'RMSprop', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                    select_loss = 'cosine', #'pinsage'
                    loss_margin = 0.25, # 0.25
                    train_pos = trainpos,
                    train_neg = trainneg,
                    val_pos = valpos)
 
nn_model.to(device)

#%% Number of learnable parameters
num_params = sum(p.numel() for p in nn_model.parameters() if p.requires_grad); print(num_params) #2672401, total: 14869969

#%% Training

lr = 1e-5; epochs = 600; loss_interval, eval_interval, emb_interval = 5, 50, 50
[train_eval, val_eval, loss_values, embs] = nn_model.train(epochs, lr, [loss_interval, eval_interval, emb_interval])

#%%
# with open('out/training.pkl', 'wb') as f: pickle.dump([embs, loss_values, train_eval, val_eval], f)
# with open('out/node_embs.pkl', 'wb') as f: pickle.dump(embs[3], f)
# with open('out/training.pkl', 'rb') as f: [embs, loss_values, train_eval, val_eval] = pickle.load(f) #480, 60, 60
# with open('out/node_embs.pkl', 'rb') as f: node_embs = pickle.load(f)

#with open('out/training.pkl', 'rb') as f: [node_embs, train_eval, val_eval, loss_values, embs] = pickle.load(f) #480, 60, 60

#%% Visualization of training
import matplotlib.pyplot as plt 
import seaborn as sns

#nn_loss, nn_hr, nn_mrr = 0.079, 27.1, 0.8 

# Plot the loss values
fig1 = plt.figure(1); plt.grid()
plt.plot(range(1,epochs+1), loss_values)
plt.xticks([1] + list(range(100, epochs+1, 100)))
fig1.suptitle('Loss value Vs. Epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss value')
#plt.axhline(y=nn_loss, color='green', linestyle='--')
#plt.text(x=0, y=nn_loss+0.02, s='NN02 ='+str(nn_loss), fontsize=12, color='green')
plt.show()

# Plot the hit-rate 
train_hr = [i for i,j in train_eval]; val_hr = [i for i,j in val_eval]
fig2 = plt.figure(2); plt.grid()
plt.xticks( list(range(len(train_hr))), [eval_interval] + list(range(eval_interval*2, eval_interval * (1+len(train_hr)), eval_interval)))
plt.plot(train_hr, label = 'train'); plt.plot(val_hr, label = 'valid'); plt.legend(loc='upper left')
fig2.suptitle('Hit-rate Vs. Epoch'); plt.xlabel('Epoch'); plt.ylabel('Hitrate')
#plt.axhline(y=nn_hr, color='green', linestyle='--')
#plt.text(x=3, y=nn_hr+0.5, s='NN02 =' + str(nn_hr), fontsize=12, color='green')

# Plot the MRR
train_mrr = [j for i,j in train_eval]; val_mrr = [j for i,j in val_eval]
fig3 = plt.figure(3); plt.grid()
plt.xticks( list(range(len(train_mrr))), [eval_interval] + list(range(eval_interval*2, eval_interval * (1+len(train_mrr)), eval_interval)))
plt.plot(train_mrr, label = 'train'); plt.plot(val_mrr, label = 'valid');  plt.legend(loc='upper left')
fig3.suptitle('MRR Vs. Epoch'); plt.xlabel('Epoch'); plt.ylabel('MRR')
#plt.axhline(y=nn_mrr, color='green', linestyle='--')
#plt.text(x=0, y=nn_mrr+0.05, s='NN02 ='+ str(nn_mrr), fontsize=12, color='green')

#fig1.savefig('diagrams/plots/gcn/lossvalues.jpg')
#fig2.savefig('diagrams/plots/gcn/hr.jpg')
#fig3.savefig('diagrams/plots/gcn/mrr.jpg')

print(max(train_eval)); print(max(val_eval))

#%% Embedding similarity distribution
import random
from scipy.stats import kurtosis 
import torch.nn.functional as F

def embplot(emb, N=2000):
  t0 = time.time()
  ind = random.sample(range(emb.shape[0]), N)
  iemb = emb[ind]
  s=[]; limit = len(iemb)-1
  
  for i,a in enumerate(iemb):
      if i == limit: break
      val = F.cosine_similarity(a[None,:], iemb[i+1:])
      s.extend(val)
  print('Time taken:', time.time()-t0)
  s = torch.stack(s)
  return s.tolist()

# Plot the curves
fig4 = plt.figure(4); plt.grid()
k = [] #kurtosis_values
for i,emb in enumerate(embs[:4]):
  e = embplot(emb, 2000); k.append(kurtosis(e)); print('Kurtosis number', i, '=',  k[i])
  sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'Epoch %d' %(emb_interval*(i+1)))

plt.xlim(-1.0,1.0); plt.legend(loc='upper center')
fig4.suptitle('Embedding similarity distribution'); plt.xlabel('Embedding cosine similarity'); plt.ylabel('Probability density of pairwise distances')

# fig4.savefig('diagrams/two_embdist.jpg')
# Kurtosis = [-1.44, -1.45, -1.51, -1.49]

#%% Test data

t0 = time.time()
with open('out/node_embs.pkl', 'rb') as f: two = pickle.load(f)
#with open('data/testpos.pkl', 'rb') as f: testpos = pickle.load(f)

eval_obj = eval.evalpipe(node_embs, K=500)
res_train = eval_obj.compute(trainpos)
res_val = eval_obj.compute(valpos)
res_test = eval_obj.compute(testpos)

print('Time taken:', time.time()-t0)
# Hitrate = 65.6, MRR = 8.6, Hitrate = 34.6 MRR = 1.4, Hitrate = 31.6 MRR = 1.5 #E1

#%%
"""
Experimental purpose only!

VERSION = "nightly" 
#param ["20200220","nightly", "xrt==1.15.0"]
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION

# imports the torch_xla package
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device() 

#--------
with open(path+'valpos.pkl', 'rb') as f: valpos = pickle.load(f)
with open(path+'testpos.pkl', 'rb') as f: testpos = pickle.load(f)
torch.save(torch.tensor(valpos), path+'valpos.pt')
torch.save(torch.tensor(testpos), path+'testpos.pt')

#--------
# 01. Embedding similarity distribution for individual data

fig1 = plt.figure(1); sns.set_style('whitegrid')

e = embplot(user_features[:,:3]); k = kurtosis(e); print('01 =', k)
sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'Numerical')

e = embplot(user_features[:,3:48]); k = kurtosis(e); print('02 =', k)
sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'Categorical')

e = embplot(user_features[:,48:]); k = kurtosis(e); print('03 =', k)
sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'S-BERT')

e = embplot(user_features); k = kurtosis(e); print('04 =', k)
sns.distplot(e, hist=False, kde = True, norm_hist=True, label = 'All')

plt.legend(loc='upper left'); plt.xlim(-0.3,1.0)
fig1.suptitle('Embedding similarity distribution'); 
plt.xlabel('Embedding cosine similarity'); plt.ylabel('Probability density of pairwise distances')
#--------
LSH:
#fig1.savefig('diagrams/emb_dist.jpg')

!pip install git+https://github.com/pixelogik/NearPy.git#egg=nearpy
#import importlib; importlib.reload(nearpy)

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
engine.store_many_vectors(user_features.numpy())

for index in range(100000):
    v = numpy.random.randn(dimension)
    engine.store_vector(v, 'data_%d' % index)

# Create random query vector
#query = user_features[2].numpy() #numpy.random.randn(dimension)

# Get nearest neighbours
#N = engine.neighbours(query)

# Get nearest neighbours
query = user_features[2].numpy() 
N = engine.neighbours(query)

#print(query)
print(N[1][0])

#--------

"""
