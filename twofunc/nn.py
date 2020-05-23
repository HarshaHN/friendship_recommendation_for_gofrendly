
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
"""
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
