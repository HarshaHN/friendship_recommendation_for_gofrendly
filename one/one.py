
#%%---------------------
"""
Date: 2 June 2020
Author: Harsha harshahn@kth.se
User profile representationSemantic similarity of user profiles, output: [hitrate, mrr]
"""

#%%-----------------------------------
""" 01. Load the feature data """
import dproc
import pandas as pd

# sqlusers = pd.read_hdf("../data/one/sqlusers.h5", key='01')     #['user_id', 'story', 'iam', 'meetfor', 'birthday', 'marital', 'children', 'lat', 'lng']
feat = pd.read_hdf("../data/one/trainfeat.h5", key='04') # ['emb', 'cat', 'num'] #dproc: preproc >> feature
[trainpos, trainneg] = dproc.dproc.loadlinks() #46581, 9986

#%%-----------------------------------
""" 02. Transform to model inputs """
import torch

categorical_data = torch.tensor(list(feat.cat), dtype=torch.float32)
numerical_data = torch.tensor(list(feat.num), dtype=torch.float32)
X = torch.cat((numerical_data, categorical_data), 1)

ids = list(feat.index)
id_idx = {id: n for n,id in enumerate(feat.index)} # dict(enumerate(feat.index))
trainpos = [(id_idx[a], id_idx[b]) for a,b in trainpos ]
trainneg = [(id_idx[a], id_idx[b]) for a,b in trainneg ]
# valpos = [(id_idx[a], id_idx[b]) for a,b in valpos]
del id_idx, numerical_data, categorical_data

#%%-------------------
""" 03. Encoder Model """
# embed = nn.Embedding(*X.shape).weight.detach()

onemodel = net( inputs = X,
                output_size = 10, #6, 10
                layers = [], #48,24,12  #48, 36, 30 
                dropout = 0.1,
                lr = 1e-2, #1e-2, 2e-3
                opt = 'RMSprop', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                cosine_lossmargin = -0.5, #-1, -0.5
                pos = trainpos, #72382
                neg = trainneg) #402761

#%%--------------------
""" 03. a. Training """
#onemodel.optimizer  = getattr(torch.optim, 'RMSprop')(onemodel.net.parameters(), 1e-2)
emb = onemodel.train(epochs = 75, lossth=0.05)
#emb[0]

#%%---------------------
""" 04. Recommendations and evaluation """

from pipe import pipeflow
onepipe = pipeflow(emb, K=500, nntype='cosine')

# df, hr, mrr
results = onepipe.dfmanip(trainpos)
#onetraindf, onetrainres = onepipe.dfmanip(trainpos)
results[0].head()

"""
import pickle
with open('../data/one/oneoutemb.pkl', 'wb') as f: pickle.dump([emb], f)
"""

#%%=================================================
"""
Date: 6 May 2020
Author: Harsha harshahn@kth.se
Graph Neural Network, output: [hitrate, mrr]
"""
import dgl
G = dproc.dproc.makedgl(num=len(ids), pos=trainpos)
g = dgl.graph(G.edges(), 'user', 'frd')
#G.readonly(True)

#%%-------------------------------------------------
# Random walk
rw = dgl.sampling.RandomWalkNeighborSampler(G=g, 
                                            random_walk_length=32, 
                                            random_walk_restart_prob=0,
                                            num_random_walks=5,
                                            num_neighbors=5,
                                            weight_column='w')
ng = rw(torch.LongTensor(g.nodes()))
ng.edata['w'] = ng.edata['w'].float()
del rw, ng.edata['_ID']

# ng.predecessors(1)
# ng.edata['w'][ng.edge_ids(*ng.in_edges(1))]

#%%----------------------
""" 01. Graph Neural Network """
#ng.is_readonly
import nn
from importlib import reload; reload(nn)

twomodel = nn.gnet( graph = ng,
                    nodeemb = X,
                    convlayers = [48, 48], # hidden, out
                    layers = 3,
                    output_size = 30,
                    dropout = 0.1,
                    lr = 1e-3,
                    opt = 'RMSprop', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                    select_loss = 'pinsage',
                    loss_margin = 2, #
                    pos = trainpos, #72382
                    neg = trainneg) #402761

#%%--------------------
""" 03. a. Training """
#twomodel.optimizer  = getattr(torch.optim, 'RMSprop')(twomodel.net.parameters(), 3e-3)
newemb = twomodel.train(epochs=50, lossth=0.05)
# list(twomodel.parameters())[0]

#%%---------------------
""" 04. Recommendations and evaluation """

from pipe import pipeflow
twopipe = pipeflow(newemb, K=500, nntype='cosine')

# df, hr, mrr
results = twopipe.dfmanip(trainpos)
#onetraindf, onetrainres = onepipe.dfmanip(trainpos)
#results[0].head(10)

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



