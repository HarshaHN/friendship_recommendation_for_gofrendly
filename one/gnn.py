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
    def __init__(self, convdim, output_size, K):
        super(PCN, self).__init__()
        [in_feats, hidden_size, out_feats] = convdim
        # kernels
        self.pinconvs = nn.ModuleList([
            pinconv.PinConv(in_feats, hidden_size, out_feats) for k in range(K) ])
        self.G = nn.Linear(out_feats, output_size)
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
class gnet(nn.Module):

    def __init__(self, graph, nodeemb, convlayers, layers, output_size, dropout, lr, opt, select_loss, loss_margin, pos, neg, fdim=24):

        super(gnet, self).__init__()
        # Graph
        self.G = graph

        # Node embeddings
        if nodeemb != None:
            nodesize, fdim = nodeemb.shape
            self.embed = nn.Embedding.from_pretrained(nodeemb)
        else:
            nodesize, fdim = self.G.number_of_nodes(), fdim
            self.embed = nn.Embedding(nodesize, fdim)
        
        # Training samples
        self.pos = torch.tensor(list(zip(*pos))) #pos = tuple(zip(pos[0],pos[1]))
        self.neg = torch.tensor(list(zip(*neg)))

        # Define the model
        self.net = PCN([fdim, *convlayers], output_size, layers) 

        # Define the optimizer
        self.optimizer = getattr(torch.optim, opt)(itertools.chain(self.net.parameters(), self.embed.parameters()), lr)

        # Loss function
        self.margin = loss_margin
        self.select_loss = select_loss
        self.loss_values = []
        
        if select_loss == 'cosine':
            self.input1 = torch.cat((self.pos[0], self.neg[0]))
            self.input2 = torch.cat((self.pos[1], self.neg[1]))
            self.target = torch.cat((torch.ones(len(self.pos[0])), torch.ones(len(self.neg[0]))*-1))
    
    # Loss funciton definitions
    def lossfunc(self, newemb, margin):
        if self.select_loss == 'cosine':
            return F.cosine_embedding_loss(newemb[self.input1], newemb[self.input2], self.target, margin)
        elif self.select_loss == 'pinsage':
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

            if epoch%10 == 0:
                print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
            self.loss_values.append(loss.item())
            if loss.item() < lossth: 
                break
        
        plt.plot(self.loss_values)
        return newemb.detach()
    
    def eval(pos):
        pass


# %%
