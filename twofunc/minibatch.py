
"""
Date: 20 May 2020
Goal: 02 Create a sample minibatch and algo 2 for minibatches. (try to do it in 3d tensors)
Author: Harsha HN harshahn@kth.se
"""
#%% ------------------------
""" 00. import libraries """
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data.utils import save_graphs, load_graphs
from pinsage import PinSageConv 

#%%-----------------------------------------
""" 01. Pinsage minibatch """


#%%-----------------------------------------
""" xx. sample minibatch """
# Graph
G = load_graphs("../data/two/dglmf_noniso.bin")[0][0] 
print('We have %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 
del G.ndata['id']
model = PinSageConv(in_features=2, out_features=2, hidden_features=2)

"""
nodeset: all the nodes in the graph
get ngbrs of each node
create embeddings and alpha
(take a subgraph for n layers)
"""

#%%----------------------------
T = dgl.DGLGraph()
T.add_nodes(5)
T.add_edges([0,0,1,2,3], [1,2,3,4,4])
T = dgl.transform.to_bidirected(T)
T.ndata['feat'] = torch.randn(5,3)
T.readonly(True)

def nxviz(G):
    import networkx as nx
    nx_G = G.to_networkx()#.to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

nxviz(T)

#%%------------------------------
""" 02. PinSAGE run """
import pinsage
from importlib import reload; reload(pinsage)

def onelayergcn(node): #nodebatch

    node.data['feat']
    # sub-graph about the node
    n = dgl.contrib.sampling.sampler.NeighborSampler(g=T, batch_size=1, expand_factor=-1, num_hops=1, neighbor_type='in', transition_prob=None, seed_nodes=node.nodes(), shuffle=False, num_workers=1, prefetch=False, add_self_loop=False)
    # get the nodeflow
    nf = n.fetch(current_nodeflow_index=0)[0]
    # copy the feature data from the parent graph
    nf.copy_from_parent()
    # get the node feature
    h_node = nf.layers[1].data['feat'][0] #nf.layers.__getitem__(1).data['feat'][0]
    # get the node feature of its neighbors
    h_ngbrs = nf.layers[0].data['feat'] #nf.layers.__getitem__(0).data['feat']
    # alpha or neighbor weights
    alpha = torch.ones_like(h_ngbrs)
    # perform pinconv and get new nodes features for the layers i+1
    node_newfeat = model(h_node, h_ngbrs, alpha, gamma='mean')

    return {'feat': node_newfeat[None]}

model = pinsage.PinSageConv(3,3,3)
T.apply_nodes(func=lambda x: onelayergcn(x))

#%%
def fake(node):
    print(node.data['feat'])
    pass #return {'x': node.data['x'] + 1}
T.apply_nodes(func=lambda x: fake(x))

#%%
"""
Assumptions: as far as the operations are sequentially arranged, they are trainable.
To-do: scale for K layers, encapsulate the vars, 

Helpers:

def update_emb(layer, a):
    out = layer.data['feat'] + a
    print(out) #nodebatch obj
    pass

nf.apply_layer(layer_id=1, func=lambda x: update_emb(x, torch.tensor([5,1])), v='__ALL__', inplace=False)

ngbrs_nodes = nf.layer_parent_nid(0)
ngbrs_nodes = T.predecessors(node)
alpha = torch.ones_like(ngbrs_nodes)


def fake(node):
    print(T) #NodeBatch
    return {'x': nodes.data['x'] + 1}

T.apply_nodes(func=lambda x: fake(x), v=[0])


T.nodes[0].data['feat']
T.readonly(True)
n = dgl.contrib.sampling.sampler.NeighborSampler(g=T, batch_size=1, expand_factor=-1, num_hops=2, neighbor_type='in', transition_prob=None, seed_nodes=torch.tensor([0]), shuffle=False, num_workers=1, prefetch=False, add_self_loop=False)
nf = n.fetch(0)[0]
nf.copy_from_parent()

nf.num_layers
nf.layer_size(0)
nf.layer_parent_nid(0)
"""


#%%-------------------------------------------
""" Utility ops"""
def clr():
    import sys
    sys.modules[__name__].__dict__.clear()