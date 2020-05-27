"""
Date: 22 May 2020
Goal: 02 Complete network, training and evaluation
Author: Harsha HN harshahn@kth.se
"""

#%%---------------
# import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import twofunc.pinconv as pinconv
import matplotlib.pyplot as plt 
import itertools
import dgl
#from importlib import reload; reload(pinconv)

#%%------------------
#Define a Graph Convolution Network
class PCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, K=2):
        super(PCN, self).__init__()
        # kernels
        self.pinconvs = nn.ModuleList([
            pinconv.PinConv(in_feats, hidden_size, out_feats) for k in range(K) ])
        self.G = nn.Linear(out_feats, out_feats)
        #self.g = nn.Parameter(torch.Tensor(1))
        #self.g = 1
        nn.init.xavier_uniform_(self.G.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.G.bias, 0)

    def forward(self, g, inputs):
        h = inputs
        for i, pconv in enumerate(self.pinconvs):
            h = pconv(g, h)
        h = F.relu(self.G(h)) #*self.g
        return h

#%%---------------------------
class gnet():
    def __init__(self, G, pos, neg):
        self.G = G #G.readonly(True)
        self.pos = torch.tensor(pos)
        self.neg = torch.tensor(neg)
        self.nodesize = self.G.number_of_nodes()
    
    def config(self, fdim, fsize, layers, opt, lr, margin):
        
        self.embed = nn.Embedding(self.nodesize, fdim) #nemb = embed.weight
        self.G.ndata['feat'] = self.embed.weight #embed.weight = G.ndata['feat']
        
        # Define the model
        fsize = [fdim]*3
        self.net = PCN(*fsize, layers)

        # Define the optimizer
        self.optimizer = getattr(torch.optim, opt)(itertools.chain(self.net.parameters(), self.embed.parameters()), lr)
        #optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr)

        # Loss function
        self.margin = margin
        self.loss_values = []

    def lossfunc(self, newemb, margin):
        p = torch.mul(newemb[self.pos[0]], newemb[self.pos[1]]).sum(1).mean()
        n = torch.mul(newemb[self.neg[0]], newemb[self.neg[1]]).sum(1).mean()
        return F.relu(n - p + margin)
    
    def train(self, epochs):
        for epoch in range(epochs):
            newemb = self.net(self.G, self.embed.weight)
            loss = self.lossfunc(newemb, self.margin)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
            self.loss_values.append(loss.item())
            if loss.item() < 0.05: break

        plt.plot(self.loss_values)
        #list(net.parameters())[0].grad
    
    def eval(pos):
        pass
