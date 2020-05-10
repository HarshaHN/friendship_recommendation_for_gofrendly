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
1. input: graph, input node features of the users (take care of isolated nodes)
2. 
"""

#%%----------------------
""" Load the network data into dgl"""
import twofunc.data as op
#from importlib import reload; reload(op)
[ids, mf] = op.loadmf()

# DGL Graph from network
import dgl
import torch

id_idx = dict(zip(ids, range(len(ids))))
[frds, G] = op.dglnx(ids, id_idx, mf)

#Save the DGL graph
from dgl.data.utils import save_graphs, load_graphs
save_graphs("./data/two/dglmf.bin", G)
G = load_graphs("./data/two/dglmf.bin")[0][0]
nx_G = G.to_networkx().to_undirected()

#%%----------------------
""" Build GCN """
import torch
import torch.nn as nn
import torch.nn.functional as F

#Add features to the DGL graph
import pandas as pd
rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04').head()
embed = nn.Embedding(1, 5)
G.ndata['feat'] = embed.weight


#%%----------------------
""" Scratch pad """
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()

X = torch.randn((28,28))
X = X.view(-1,28*28)
output = net(X)
    
    

#%%----------------------




#%%----------------------
""" Import libraries """





#%%----------------------
""" Import libraries """

#%%--------------------------------------------
""" Recsys for all users 
[auc, hitrate, mrr] = recsysone(model, *(df, links))
"""

#%%-------------------------------------------
""" Utility ops"""
def clr():
    import sys
    sys.modules[__name__].__dict__.clear()
