#%%---------------
# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import pinconv
#from pinconv import PinConv
import matplotlib.pyplot as plt 
import itertools
import dgl


#%%-----------------------------------------
from importlib import reload; reload(pinconv)

#g.add_edges(g.nodes(), g.nodes())

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

#%%------------------
#Define a Graph Convolution Network
class PCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(PCN, self).__init__()
        self.conv1 = pinconv.PinConv(in_feats, hidden_size, out_feats)
        self.conv2 = pinconv.PinConv(in_feats, hidden_size, out_feats)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = self.conv2(g, h)
        return h

net = PCN(5,5,5)

T = getT()
embed = nn.Embedding(T.number_of_nodes(), 5)
T.ndata['feat'] = embed.weight
inputs = embed.weight

pos = torch.tensor([[0], [1]], requires_grad=False)
neg = torch.tensor([[0], [3]], requires_grad=False)
# Define the loss func
def lossfunc(newemb):
    p = torch.mul(newemb[pos[0]], newemb[pos[1]]).sum(1).mean()
    n = torch.mul(newemb[neg[0]], newemb[neg[1]]).sum(1).mean()
    margin = 1.0
    return F.relu(n - p + margin)

optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=2e-4)
loss_values = []

#%%---------------------------------

for epoch in range(200):
    logits = net(T, inputs)
    # we save the logits for visualization later
    # all_logits.append(logits.detach())
    #logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    #loss = F.nll_loss(logp[labeled_nodes], labels)
    loss = lossfunc(logits)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())
    if loss.item() < 0.1: break
    #print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
plt.plot(loss_values)

#list(net.parameters())[0].grad

#%%-----------
import sys
sys.modules[__name__].__dict__.clear()


#%%

















"""
#%%---------------------
""" Import libraries """
from pinsage import PinSage
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import tqdm
#from rec.utils import cuda
from dgl import DGLGraph
from dgl.data.utils import save_graphs, load_graphs

#%%---------------------
""" Main """
# Load the graph
g = load_graphs("./data/two/dglmf_noniso.bin")[0][0]
print('We have %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 

# Add mode features
embed = nn.Embedding(G.number_of_nodes(), 2)
G.ndata['feat'] = embed.weight

#%%
# Hyper parameters
n_hidden = 1
n_layers = 1 #args.layers
batch_size = 256
margin = 0.9
opt = 'SGD'
loss = 'hinge'
lr = 0.01
n_negs = 0 #args.n_negs
hard_neg_prob = 0 #args.hard_neg_prob

# Loss func
loss_func = {
        'hinge': lambda diff: (diff + margin).clamp(min=0).mean(),
        'bpr': lambda diff: (1 - torch.sigmoid(-diff)).mean(),
        }

# Define the optimizer
opt = getattr(torch.optim, opt)(model.parameters(), lr=lr, momentum=0.9)

# Model declaration
model = PinSage(
                num_nodes = G.number_of_nodes(),
                feature_sizes = [2]*2, #[n_hidden] * (n_layers + 1),
                T = 3,
                restart_prob = 0.5,
                max_nodes = 3,
                use_feature = False,
                G=G,
                )

h_new = model(G, nodeset)

#%%----------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

#Define a Graph Convolution Network
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = F.softmax(h, 1)
        return h

    #train
    def traingnn(self, frds, embed, inputs):
        import itertools
        label1 = torch.tensor([item[0] for item in frds])
        label2 = torch.tensor([item[1] for item in frds])
        target = torch.ones(len(frds))
        criterion = nn.CosineEmbeddingLoss()

        optimizer = torch.optim.Adam(itertools.chain(self.net.parameters(), embed.parameters()), lr=0.01)
        all_logits = []
        for epoch in range(50):
            logits = self.net(G, inputs)
            # we save the logits for visualization later
            all_logits.append(logits.detach())
            #logp = F.log_softmax(logits, 1)
            loss = criterion( logits[label1],
                            logits[label2],
                            target )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


"""

























#%% ------------------------
""" """
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from pinsage import PinSage
#import randomwalk

#%%----------------------------
from dgl.data.utils import save_graphs, load_graphs
G = load_graphs("./data/two/dglmf_noniso.bin")[0][0]
print('We have %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 




#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class PinSageConv(nn.Module):

    # feature size
    def __init__(self, in_features, out_features, hidden_features):
        super(PinSageConv, self).__init__()

        # feature size, kernel, params
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.Q = nn.Linear(in_features, hidden_features)
        self.W = nn.Linear(in_features + hidden_features, out_features)

        nn.init.xavier_uniform_(self.Q.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, h, node, nb_nodes, alpha, gamma):
        """
         h: node emb ( num_neighbors, in_feature),
         nodeset: node ids (1, ),
         nb_nodes: neighbors (1, num_neighbors),
         alpha(nb_weights): (1, num_neighbors),
         gamma(mean): simple mean op
         return: new node emb # (1, out_features)
        """

        #Line 1
        h_neighbors = F.relu(self.Q(h[nb_nodes])) #(num_neighbors, hidden_features)
        #n_u = gamma(alpha(h_neighbors))
        temp = h_neighbors * alpha[nb_nodes, None]
        n_u = temp.sum(0) / alpha[nb_nodes].sum(-1)

        #Line 2
        znu = torch.cat([h[node], n_u], -1)
        zu = F.relu(self.W(znu)) #(out_features)

        #Line 3
        b = zu.norm(dim=-1, keepdim=True)
        b = torch.where(b == 0, torch.ones_like(b), b)
        h_new = zu / b
    
        return h_new

#%%
import dgl

T = dgl.DGLGraph()
T.add_nodes(3)
T.add_edges(0, [1,2])

G.ndata['feat'] = torch.ones(3,2)
h = torch.ones(3,2)

#%%
model = PinSageConv(2,2,4)
alpha= torch.tensor([1,2,3])
#%%
hi = model( h, node=0, nb_nodes=[1,2], alpha = alpha, gamma=0)


# %%












#%%-------------------------
""" """
def get_embeddings(h, nodeset):
    return h[nodeset]

def safediv(a, b):
    b = torch.where(b == 0, torch.ones_like(b), b)
    return a / b

def init_weight(w, func_name, nonlinearity):
    getattr(nn.init, func_name)(w, gain=nn.init.calculate_gain(nonlinearity))

def init_bias(w):
    nn.init.constant_(w, 0)

#%%-----------------------------
class PinSageConv(nn.Module):

    # feature size
    def __init__(self, in_features, out_features, hidden_features): # 3,6,3
        super(PinSageConv, self).__init__()
        # feature size, kernels, params init

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.Q = nn.Linear(in_features, hidden_features)
        self.W = nn.Linear(in_features + hidden_features, out_features)

        init_weight(self.Q.weight, 'xavier_uniform_', 'leaky_relu')
        init_weight(self.W.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(self.Q.bias)
        init_bias(self.W.bias)

    def forward(self, h, nodeset, nb_nodes, nb_weights):
        # h, u & N(u), alpha(nb_weights), gamma(mean)
        '''
        h: node embeddings (num_total_nodes, in_features), or a container
           of the node embeddings (for distributed computing)
        nodeset: node IDs in this minibatch (num_nodes, )
        nb_nodes: neighbor node IDs of each node in nodeset (num_nodes, num_neighbors)
        nb_weights: weight of each neighbor node (num_nodes, num_neighbors)
        return: new node embeddings (num_nodes, out_features)
        '''
        # Gather
        num_nodes, T = nb_nodes.shape #(num_nodes, num_neighbors)
        h_neighbors = get_embeddings(h, nb_nodes.view(-1)).view(num_nodes, T, self.in_features)
        
        # Line 1
        h_neighbors = F.leaky_relu(self.Q(h_neighbors))
        h_agg = safediv( # (num_nodes, in_features)
                (nb_weights[:, :, None] * h_neighbors).sum(1), #alpha
                nb_weights.sum(1, keepdim=True)) #gamma
        
        # Line 2
        h_nodeset = get_embeddings(h, nodeset)  # (num_nodes, in_features)
        h_concat = torch.cat([h_nodeset, h_agg], 1)
        h_new = F.leaky_relu(self.W(h_concat)) # (num_nodes, in_features + hidden_features)

        # Line 3
        h_new = safediv(h_new, h_new.norm(dim=1, keepdim=True))
        # (num_nodes, out_features)
        return h_new


#%%-----------------------------------------
def create_embeddings(n_nodes, n_features):
    return nn.Parameter(torch.randn(n_nodes, n_features))

def mix_embeddings(h, ndata, emb, proj):
    '''Combine node-specific trainable embedding ``h`` with categorical inputs
    (projected by ``emb``) and numeric inputs (projected by ``proj``).
    '''
    e = []
    for key, value in ndata.items():
        if value.dtype == torch.int64:
            e.append(emb[key](value))
        elif value.dtype == torch.float32:
            e.append(proj[key](value))
    return h + torch.stack(e, 0).sum(0)
    #(self.h, G.ndata, self.emb, self.proj)

def get_embeddings(h, nodeset):
    return h[nodeset]

def put_embeddings(h, nodeset, new_embeddings):
    n_nodes = nodeset.shape[0]
    n_features = h.shape[1]
    return h.scatter(0, nodeset[:, None].expand(n_nodes, n_features), new_embeddings)

def safediv(a, b):
    b = torch.where(b == 0, torch.ones_like(b), b)
    return a / b

def init_weight(w, func_name, nonlinearity):
    getattr(nn.init, func_name)(w, gain=nn.init.calculate_gain(nonlinearity))

def init_bias(w):
    nn.init.constant_(w, 0)

class PinSageConv(nn.Module):
    def __init__(self, in_features, out_features, hidden_features): # 3,6,3
        super(PinSageConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.Q = nn.Linear(in_features, hidden_features)
        self.W = nn.Linear(in_features + hidden_features, out_features)

        init_weight(self.Q.weight, 'xavier_uniform_', 'leaky_relu')
        init_weight(self.W.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(self.Q.bias)
        init_bias(self.W.bias)


    def forward(self, h, nodeset, nb_nodes, nb_weights):
        '''
        h: node embeddings (num_total_nodes, in_features), or a container
           of the node embeddings (for distributed computing)
        nodeset: node IDs in this minibatch (num_nodes, )
        nb_nodes: neighbor node IDs of each node in nodeset (num_nodes, num_neighbors)
        nb_weights: weight of each neighbor node (num_nodes, num_neighbors)
        return: new node embeddings (num_nodes, out_features)
        '''
        num_nodes, T = nb_nodes.shape #(num_nodes, num_neighbors)
        h_neighbors = get_embeddings(h, nb_nodes.view(-1)).view(num_nodes, T, self.in_features)

        h_neighbors = F.leaky_relu(self.Q(h_neighbors))
        h_agg = safediv(
                (nb_weights[:, :, None] * h_neighbors).sum(1),
                nb_weights.sum(1, keepdim=True))

        h_nodeset = get_embeddings(h, nodeset)  # (num_nodes, in_features)
        h_concat = torch.cat([h_nodeset, h_agg], 1)
        h_new = F.leaky_relu(self.W(h_concat))
        h_new = safediv(h_new, h_new.norm(dim=1, keepdim=True))

        return h_new

#%%

class PinSage(nn.Module):
    '''
    Completes a multi-layer PinSage convolution
    G: DGLGraph
    feature_sizes: the dimensionality of input/hidden/output features
    T: number of neighbors we pick for each node
    restart_prob: restart probability
    max_nodes: max number of nodes visited for each seed
    '''
    def __init__(self, num_nodes, feature_sizes, T, restart_prob, max_nodes,
                 use_feature=False, G=None):
        super(PinSage, self).__init__()

        self.T = T
        self.restart_prob = restart_prob
        self.max_nodes = max_nodes

        self.in_features = feature_sizes[0]
        self.out_features = feature_sizes[-1]
        self.n_layers = len(feature_sizes) - 1

        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(PinSageConv(
                feature_sizes[i], feature_sizes[i+1], feature_sizes[i+1]))

        self.h = create_embeddings(num_nodes, self.in_features)
        self.use_feature = use_feature

        if use_feature:
            self.emb = nn.ModuleDict()
            self.proj = nn.ModuleDict()

            for key, scheme in G.node_attr_schemes().items():
                if scheme.dtype == torch.int64:
                    self.emb[key] = nn.Embedding(
                            G.ndata[key].max().item() + 1,
                            self.in_features,
                            padding_idx=0)
                elif scheme.dtype == torch.float32:
                    self.proj[key] = nn.Sequential(
                            nn.Linear(scheme.shape[0], self.in_features),
                            nn.LeakyReLU(),
                            )

    def forward(self, G, nodeset):
        '''
        Given a complete embedding matrix h and a list of node IDs, return
        the output embeddings of these node IDs.

        nodeset: node IDs in this minibatch (num_nodes,)
        return: new node embeddings (num_nodes, out_features)
        '''
        if self.use_feature:
            h = mix_embeddings(self.h, G.ndata, self.emb, self.proj)
        else:
            h = self.h

        nodeflow = randomwalk.random_walk_nodeflow(
                G, nodeset, self.n_layers, self.restart_prob, self.max_nodes, self.T)

        for i, (nodeset, nb_weights, nb_nodes) in enumerate(nodeflow):
            new_embeddings = self.convs[i](h, nodeset, nb_nodes, nb_weights)
            h = put_embeddings(h, nodeset, new_embeddings)

        h_new = get_embeddings(h, nodeset)
        return h_new

