"""
Date: 6 May 2020
Author: Harsha harshahn@kth.se
Graph Neural Network, output: [auroc, hitrate, mrr]

Referencs:
PinSAGE, GCN, GNN, 
https://github.com/rusty1s/pytorch_geometric
https://github.com/dmlc/dgl
https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/link_predict.py#L34

"""

"""
c. module for pinsage.
d. module for training.
e. experiments for optimization.
"""

#%%----------------------
""" 01. Load the network data into dgl """
import twofunc.data as op
#from importlib import reload; reload(op)
[ids, [trainids, trainmf]] = op.loadmf()

# DGL Graph from network
import dgl
import torch

id_idx = dict(zip(trainids, range(len(trainids))))
[trainfrds, G] = op.dglnx(trainids, id_idx, trainmf)

#%%-------------------------------
#Save the DGL graph
from dgl.data.utils import save_graphs, load_graphs
#save_graphs("./data/two/dglmf_noniso.bin", G) #dglmf
G = load_graphs("./data/two/dglmf_noniso.bin")[0][0]
print('We have %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 
#nxviz(G)

#%%----------------------
""" 02. Assign features to nodes or edges """
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add features to each node G.subgraph(1)
embed = nn.Embedding(G.number_of_nodes(), 2)
G.ndata['feat'] = embed.weight #op.getrawemb(G.number_of_nodes())

#%%----------------------
""" Graph Neural Network """
import twofunc.nn as nnop
net = nnop.GCN(2, 3, 2)

#%%----------------------
""" Data preparation and initialization """
inputs = G.ndata['feat']

#%%----------------------
""" Train and visualize """
net.traingnn(trainfrds, embed, inputs)

#%%----------------------
"""  """

#%%----------------------
""" Load validation dataset """
import twofunc.data as op

# trainmf, trainids, id_idx >> deltavalmf(id_idx) which are in trainids
dvalfrds = op.deltamf(trainmf, trainids, id_idx)
del trainmf

#%%--------------------------------------------
""" Recsys for all users """
import twofunc.recs as gnn
# from importlib import reload; reload(gnn)

# model = load('./data/model/gnn.pkl')
recsys = gnn.recsystwo(len(trainids))
recsdf = recsys.dfmanip(dvalfrds, 10)
[hitrate, mrr] = recsys.eval()

#%%-------------------------------------------
""" Utility ops"""
def clr():
    import sys
    sys.modules[__name__].__dict__.clear()

# %%
