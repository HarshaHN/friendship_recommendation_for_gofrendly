"""
Date: 22 May 2020
Goal: 02 Complete network and training.
Author: Harsha HN harshahn@kth.se
"""
#%%---------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import pinconv
import matplotlib.pyplot as plt 
import itertools
import dgl
# from importlib import reload; reload(pinconv)

#%%------------------
#Define a Graph Convolution Network
class PCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, K=2):
        super(PCN, self).__init__()
        # kernels
        self.pinconvs = nn.ModuleList([
            pinconv.PinConv(in_feats, hidden_size, out_feats) for k in range(K) ])
        self.G = nn.Linear(out_feats, out_feats)
        self.g = nn.Parameter(torch.Tensor(1))
        
        nn.init.xavier_uniform_(self.G.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.G.bias, 0)
        nn.init.constant_(self.g, 1)
        
    def forward(self, g, inputs):
        h = inputs
        for i, pconv in enumerate(self.pinconvs):
            h = pconv(g, h)
        h = F.relu(self.G(h)) * self.g
        return h

#%%---------------------------
class gnet():
    def __init__(self, G, pos, neg):
        self.G = G #self.G.readonly(True)
        self.pos = torch.tensor(pos)
        self.neg = torch.tensor(neg)
        self.nodesize = self.G.number_of_nodes()
    
    def config(self, fdim, fsize, layers, opt, lr, margin, selectLoss='cosine', embflag=False, nodefeat=None):
            
        if embflag:
            self.embed = nn.Embedding.from_pretrained(nodefeat)
            fdim = nodefeat.shape[1]
        else: 
            self.embed = nn.Embedding(self.nodesize, fdim)
        self.G.ndata['feat'] = self.embed.weight

        # Define the model
        fsize = [fdim]*3
        self.net = PCN(*fsize, layers)

        # Define the optimizer
        self.optimizer = getattr(torch.optim, opt)(itertools.chain(self.net.parameters(), self.embed.parameters()), lr)

        # Loss function
        self.margin = margin
        self.selectLoss = selectLoss
        self.loss_values = []
        
        if selectLoss == 'cosine':
            self.input1 = torch.cat((self.pos[0], self.neg[0]))
            self.input2 = torch.cat((self.pos[1], self.neg[1]))
            self.target = torch.cat((torch.ones(len(self.pos[0])), torch.ones(len(self.neg[0]))*-1))

    def lossfunc(self, newemb, margin):
        if self.selectLoss == 'cosine':
            return F.cosine_embedding_loss(newemb[self.input1], newemb[self.input2], self.target, margin)
        elif self.selectLoss == 'pinsage':
            p = torch.mul(newemb[self.pos[0]], newemb[self.pos[1]]).sum(1).mean()
            n = torch.mul(newemb[self.neg[0]], newemb[self.neg[1]]).sum(1).mean()
            return F.relu(n - p + margin)
            
    def train(self, epochs, lossth):
        for epoch in range(epochs):
            newemb = self.net(self.G, self.embed.weight)
            loss = self.lossfunc(newemb, self.margin)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
            self.loss_values.append(loss.item())
            if loss.item() < lossth: 
                break

        plt.plot(self.loss_values)
        #list(net.parameters())[0].grad
    
    def eval(pos):
        pass
