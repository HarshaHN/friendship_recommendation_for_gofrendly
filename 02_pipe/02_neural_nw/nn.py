"""
File: nn.py
Date: 22 May 2020
Author: Harsha HN harshahn@kth.se
PinSAGE based neural network and training
Developed using PyTorch and DGL libraries
"""

#%%
import dgl
import eval
import torch
import pinconv
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt 

#%% Define a Graph Convolution Network

class PCN(nn.Module):
    def __init__(self, convdim, output_size, dropout=0):
        super(PCN, self).__init__()

        # kernels
        self.pinconvs = nn.ModuleList([pinconv.PinConv(*i, dropout) for i in convdim])
        self.G = nn.Linear(convdim[-1][-1], output_size)
        self.g = nn.Parameter(torch.Tensor(1))
        
        # weights initialization
        nn.init.xavier_uniform_(self.G.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.G.bias, 0)
        nn.init.constant_(self.g, 1)
        
        # norms
        #self.bnorm_in = nn.BatchNorm1d(convdim[0][0])
        self.bnorm_out = nn.BatchNorm1d(output_size)
        self.bnorm = nn.BatchNorm1d(output_size)
        
        # dropouts
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, g, h):
        """
        g : social network graph
        h : node features
        """
        #h = self.bnorm_in(h)  
        for i, pconv in enumerate(self.pinconvs):
            h = pconv(g, h)

        h = self.g * self.dropout_out(self.bnorm_out(F.relu(self.G(h))))
        h = self.bnorm(h)
 
        return h

#%% Training phase

class gnet(nn.Module):

    def __init__(self, graph, node_features, convlayers, output_size, dropout, lr, opt, select_loss, loss_margin, train_pos, train_neg, val_pos):
      """
      graph: social network graph
      node_features: node features
      convlayers: convolution layers with feature dimensions
      output_size: output embedding size
      dropout, lr: learning rate, opt: optimizer, select_loss: loss type, loss_margin,
      train_pos: positive links, train_neg: negative links, val_pos: positive link for validation
      """
      super(gnet, self).__init__()
      
      # Device
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      
      # Graph
      self.G = graph

      # Features
      nodesize, fdim = node_features.shape
      self.embed = nn.Embedding.from_pretrained(node_features)
      print('Embeddings have been loaded!')

      # Define the model
      self.dropout = dropout
      self.net = PCN([ [fdim, *convlayers[0]], *convlayers[1:] ], output_size, self.dropout) 
      
      # Define the optimizer
      self.lr = lr
      self.optimizer = getattr(torch.optim, opt)(self.net.parameters(), self.lr)
      # self.optimizer = getattr(torch.optim, opt)(itertools.chain(self.net.parameters(), self.embed.parameters()), lr)

      # Training samples
      self.pos = torch.tensor(list(zip(*train_pos))) #pos = tuple(zip(pos[0],pos[1]))
      self.neg = torch.tensor(list(zip(*train_neg)))
      self.val_pos = val_pos
      
      self.input1 = torch.cat((self.pos[0], self.neg[0]))
      self.input2 = torch.cat((self.pos[1], self.neg[1]))
      self.target = torch.cat((torch.ones(len(self.pos[0])), torch.ones(len(self.neg[0]))*-1)).to(self.device)
      
      # Loss function
      self.select_loss = select_loss
      self.margin = loss_margin
      self.loss_values = []
      # print('Non-trained loss =', self.lossfunc(self.embed.weight, self.margin).item()])
        
    def lossfunc(self, node_emb, margin):
        return F.cosine_embedding_loss(node_emb[self.input1], node_emb[self.input2], self.target, margin)

    def train(self, epochs, intervals):
      train_eval, val_eval, embs = [], [], []
      [loss_interval, eval_interval, emb_interval] = intervals

      for i in range(1,epochs+1):
          node_emb = self.net(self.G, self.embed.weight)
          loss = self.lossfunc(node_emb, self.margin)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          if i%loss_interval == 0:
            print('Epoch %d | Loss: %.4f' % (i, loss.item()))
          if i%eval_interval == 0:
            print('Grad / Weights =', (list(self.net.parameters())[1].grad / list(self.net.parameters())[1]).mean().item())
            
            # Evaluation on train data
            eval_obj = eval.evalpipe(node_emb, K=500)
            train_eval.append(eval_obj.compute(self.train_pos))
            
            # Evaluation on validation data
            eval_obj = eval.evalpipe(node_emb, K=500)
            val_eval.append(eval_obj.compute(self.val_pos))

          if i%emb_interval == 0:
            embs.append(node_emb)            

          self.loss_values.append(loss.item())
          if loss.item() < 0.01: 
            break

      return [node_emb.detach(), train_eval, val_eval, self.loss_values, embs]


#%%
"""
Experimental purpose only!

# Triplet Loss function
def lossfunc(self, node_emb, margin):
    if self.select_loss == 'cosine':
        return F.cosine_embedding_loss(node_emb[self.input1], node_emb[self.input2], self.target, self.margin)
    elif self.select_loss == 'pinsage':
        p = torch.mul(node_emb[self.pos[0]], node_emb[self.pos[1]]).mean(1).mean()
        n = torch.mul(node_emb[self.neg[0]], node_emb[self.neg[1]]).mean(1).mean()
        return F.relu(n - p + margin)

#from importlib import reload; reload(pinconv)
#import itertools

"""
