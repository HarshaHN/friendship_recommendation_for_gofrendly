
"""
Date: 26 May 2020
Goal: 02 Create a sample minibatch and algo 2 for minibatches. (try to do it in 3d tensors)
Author: Harsha HN harshahn@kth.se
"""

#%%---------------
# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pinconv
import matplotlib.pyplot as plt 
import itertools
import dgl
from dgl.data.utils import save_graphs, load_graphs
from importlib import reload; reload(pinconv)

#%%-----------------------------------------

def nxviz(G):
    import networkx as nx
    nx_G = G.to_networkx()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

def getG():
    #g.add_edges(g.nodes(), g.nodes())
    G = load_graphs("../data/two/dglmf_noniso.bin")[0][0] 
    print('We have %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 
    del G.ndata['id']
    G = dgl.transform.to_bidirected(G)
    G.readonly(True)
    return G

def getT():
    T = dgl.DGLGraph()
    T.add_nodes(5)
    T.add_edges([0,0,1,2,3], [1,2,3,4,4])
    T = dgl.transform.to_bidirected(T)
    T.readonly(True)
    return T


def getlinks():
    pos = torch.tensor([[0], [1]], requires_grad=False) #mf
    neg = torch.tensor([[0], [3]], requires_grad=False) #bf
    return [pos, neg]

#nxviz(getT())

#%%------------------
#Define a Graph Convolution Network
class PCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, K=2):
        super(PCN, self).__init__()
        # kernels
        self.pinconvs = nn.ModuleList([
            pinconv.PinConv(in_feats, hidden_size, out_feats) for k in range(K) ])
        self.G = nn.Linear(out_feats, out_feats)
        nn.init.xavier_uniform_(self.G.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.G.bias, 0)

    def forward(self, g, inputs):
        h = inputs
        for i, pconv in enumerate(self.pinconvs):
            h = pconv(g, h)
        h = F.relu(self.G(h))
        return h

#%%-------------------------
#Hyper parameters
fdim = 8
fsize = [fdim]*3
layers = 2
lr = 5e-4
epochs = 100*2
margin = .0 #loss

# Get the graph
G = getT() # getG()
embed = nn.Embedding(G.number_of_nodes(), fdim)
nemb = embed.weight
G.ndata['feat'] = embed.weight #embed.weight = G.ndata['feat']

# Define the model
net = PCN(*fsize, layers)

# Get links
[pos, neg] = getlinks()

# Define the loss func
def lossfunc(newemb, margin):
    p = torch.mul(newemb[pos[0]], newemb[pos[1]]).sum(1).mean()
    n = torch.mul(newemb[neg[0]], newemb[neg[1]]).sum(1).mean()
    return F.relu(n - p + margin)

# Define the optimizer
optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr)
loss_values = []

#%%---------------------------------
epochs = 100*5
for epoch in range(epochs):
    newemb = net(G, embed.weight)
    loss = lossfunc(newemb, margin)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
    loss_values.append(loss.item())
    if loss.item() < 0.05: break
    
plt.plot(loss_values)
#list(net.parameters())[0].grad

#%%-----------
import sys
sys.modules[__name__].__dict__.clear()

