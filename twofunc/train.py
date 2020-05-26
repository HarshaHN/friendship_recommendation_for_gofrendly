
"""
Date: 26 May 2020
Goal: 02 Create a sample minibatch and algo 2 for minibatches. (try to do it in 3d tensors)
Author: Harsha HN harshahn@kth.se
"""

#%% ------------------------
""" 00. import libraries """
import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data.utils import save_graphs, load_graphs
from pinsage import PinSageConv 
import matplotlib.pyplot as plt 

#%%-----------------------------------------
""" 01. Load the graph data """
"""
G = load_graphs("../data/two/dglmf_noniso.bin")[0][0] 
print('We have %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 
del G.ndata['id']
model = PinSageConv(in_features=2, out_features=2, hidden_features=2)

"""
"""01. Dummy graph data"""
def getT():
    T = dgl.DGLGraph()
    T.add_nodes(5)
    T.add_edges([0,0,1,2,3], [1,2,3,4,4])
    T = dgl.transform.to_bidirected(T)
    T.readonly(True)
    return T

def nxviz(G):
    import networkx as nx
    nx_G = G.to_networkx()#.to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

#nxviz(getT())



#%%-------------------------------
T = getT()
K = 2
S = list()
S = S.extend([list()*K, list(T.nodes())])












#%%------------------------------
""" 02. PinSAGE minibatch """

import pinsage
from importlib import reload; reload(pinsage)
class minibatch:
    def __init__(self, T, inputs):
        self.T = T
        self.T.ndata['feat'] = inputs
        fsize = [inputs.shape[1]] * 3
        self.model = pinsage.PinSageConv(*fsize)

    def run(self):
        for node in self.T.nodes():
            self.T.apply_nodes(func=self.onelayergcn, v=[node])
        return self.T.ndata['feat']

    def onelayergcn(self, node): #nodebatch
        # sub-graph about the node
        n = dgl.contrib.sampling.sampler.NeighborSampler(g=self.T, batch_size=1, expand_factor=-1, num_hops=1, neighbor_type='in', transition_prob=None, seed_nodes=node.nodes(), shuffle=False, num_workers=1, prefetch=False, add_self_loop=False)
        # get the nodeflow
        nf = n.fetch(current_nodeflow_index=0)[0]
        # copy the feature data from the parent graph
        nf.copy_from_parent()
        # get the node feature
        h_node = nf.layers[1].data['feat'][0]
        # get the node feature of its neighbors
        h_ngbrs = nf.layers[0].data['feat']
        # alpha or neighbor weights
        alpha = torch.ones_like(h_ngbrs)
        # perform pinconv and get new nodes features for the layers i+1
        node_newfeat = self.model(h_node, h_ngbrs, alpha, gamma='mean')
        return {'feat': node_newfeat[None]}

#%%------------------------------
""" 02. PinSAGE train """
# positive and negative links ([0,0,1,2,3], [1,2,3,4,4])
pos = torch.tensor([[0,1,2], [1,3,4]], requires_grad=False)
neg = torch.tensor([[0,3], [2,4]], requires_grad=False)

# Define the model
fsize = 20
mb = minibatch(getT(), inputs=nn.Embedding(5, fsize).weight) 

# Define the optimizer
optimizer = torch.optim.Adam(mb.model.parameters(), lr=1e-5)

# Define the loss func
def lossfunc(newemb):
    p = torch.mul(newemb[pos[0]], newemb[pos[1]]).sum(1).mean()
    n = torch.mul(newemb[neg[0]], newemb[neg[1]]).sum(1).mean()
    margin = 1.0
    return F.relu(n-p + margin)#.clamp(min=0) - p + margin

"""
import time
print(list(mb.model.parameters())[0], '\n')
print('Training begins ...')
print('Loss: ', lossfunc(mb.T.srcdata['feat']).item())
print('--> mb.T:', mb.T.srcdata['feat'], '\n')
"""
#%%
loss_values = [0]
for epoch in range(150):
    # Generate new embeddings
    newemb = mb.run()
    # Compute the loss
    loss = lossfunc(newemb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    newemb.detach_()
    
    loss_values.append(loss.item())

    #print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    #time.sleep(0.2)
    #print(mb.T.srcdata['feat'], '\n')
    #print('Grad: \n', list(mb.model.parameters())[0].grad)
    #print('--> newemb:', newemb)

plt.plot(loss_values)

#%%
"""    
    logits = net(G, inputs)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

###########################################3333

Assumptions: as far as the operations are sequentially arranged, they are trainable.
To-do: scale for K layers, encapsulate the vars, 

Helpers:
"""

#%%-------------------------------------------
""" Utility ops"""
def clr():
    import sys
    sys.modules[__name__].__dict__.clear()


