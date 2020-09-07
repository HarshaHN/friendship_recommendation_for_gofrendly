"""
Date: 2 June 2020
Goal: 01 Neural network and training.
Author: Harsha HN harshahn@kth.se
"""
#%%---------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import itertools
import pipe; import importlib; importlib.reload(pipe)

#Define the neural network
class neuralnet(nn.Module):
    def __init__(self, input_size, output_size, layers, dropout=0.0):
        super().__init__()
        
        # kernels
        self.bnorm = nn.BatchNorm1d(input_size)
        all_layers = []
        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(dropout))
            input_size = i
        
        if len(layers) == 0: layers = [input_size]
        
        all_layers.extend([ nn.Linear(layers[-1], output_size),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm1d(output_size)])
        self.layers = nn.Sequential(*all_layers)

        # initializations
        self.layers.apply(self._init_weights)
    
    def forward(self, x):
        x = self.bnorm(x)
        #x = torch.cat([x_categorical, x_numerical], 1)
        x = self.layers(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.0)

    #%%---------------------------
class net(nn.Module):
   
    def __init__(self, inputs, output_size, layers, dropout, lr, opt, cosine_lossmargin, pos, neg, val_pos):
        super(net, self).__init__()
        
        # Device
        self.device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
        
        # Features
        self.embed = nn.Embedding.from_pretrained(inputs)
        users, input_size = inputs.shape
      
        # Define the model
        self.net = neuralnet(input_size, output_size, layers, dropout)

        # Define the optimizer
        self.lr = lr
        self.optimizer = getattr(torch.optim, opt)(self.net.parameters(), self.lr)

        # Training samples
        self.train_pos = pos
        self.val_pos = val_pos
        self.pos = torch.tensor(list(zip(*pos))) #pos = tuple(zip(pos[0],pos[1]))
        self.neg = torch.tensor(list(zip(*neg)))
        
        self.input1 = torch.cat((self.pos[0], self.neg[0]))
        self.input2 = torch.cat((self.pos[1], self.neg[1]))
        self.target = torch.cat((torch.ones(len(self.pos[0])), torch.ones(len(self.neg[0]))*-1)).to(self.device)

        # Loss function
        self.margin = cosine_lossmargin
        self.loss_values = []
        print('Non-trained loss =', self.lossfunc(self.embed.weight, self.margin).item())
        
    def lossfunc(self, newemb, margin):
        return F.cosine_embedding_loss(newemb[self.input1], newemb[self.input2], self.target, margin)

    def train(self, epochs, intervals):
      train_eval, val_eval, embs = [], [], []
      [loss_interval, eval_interval, emb_interval] = intervals

      for i in range(1,epochs+1):
          newemb = self.net(self.embed.weight)
          loss = self.lossfunc(newemb, self.margin)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          if i%loss_interval == 0:
            print('Epoch %d | Loss: %.4f' % (i, loss.item()))
          if i%eval_interval == 0:
            print('Grad / Weights =', (list(self.net.parameters())[1].grad / list(self.net.parameters())[1]).mean().item())
            perf_pipe = pipe.pipeflow(newemb, K=500)
            train_eval.append(perf_pipe.dfmanip(self.train_pos))
            #perf_pipe = pipe.pipeflow(newemb[idx], K=500)
            val_eval.append(perf_pipe.dfmanip(self.val_pos))

          if i%emb_interval == 0:
            embs.append(newemb)            

          self.loss_values.append(loss.item())
          if loss.item() < 0.01: 
            break

      return [newemb.detach(), train_eval, val_eval, self.loss_values, embs]

#%%
"""
Date: 22 May 2020
Goal: 02 PinSAGE based GNN.
Author: Harsha HN harshahn@kth.se
"""

#%%---------------
import dgl
import torch
import pinconv
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 
from importlib import reload; reload(pinconv)
#import itertools

#Define a Graph Convolution Network
class PCN(nn.Module):
    def __init__(self, convdim, output_size, dropout=0):
        super(PCN, self).__init__()

        # kernels
        self.pinconvs = nn.ModuleList([pinconv.PinConv(*i, dropout) for i in convdim])
        self.G = nn.Linear(convdim[-1][-1], output_size)
        self.g = nn.Parameter(torch.Tensor(1))
        
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
        #h = self.bnorm_in(h)  
        for i, pconv in enumerate(self.pinconvs):
            h = pconv(g, h)

        h = self.g * self.dropout_out(self.bnorm_out(F.relu(self.G(h))))
        h = self.bnorm(h)
 
        return h

##%%---------------------------
class gnet(nn.Module):

    def __init__(self, graph, nodeemb, convlayers, output_size, dropout, lr, opt, select_loss, loss_margin, pos, neg, train_pos, val_pos, idx):
        super(gnet, self).__init__()
        
        # Device
        self.device = torch.device('cuda')# if torch.cuda.is_available() else 'cpu')
        
        # Graph
        self.G = graph

        # Features
        nodesize, fdim = nodeemb.shape
        self.embed = nn.Embedding.from_pretrained(nodeemb)
        print('Embeddings have been loaded!')

        # Define the model
        self.dropout = dropout
        self.net = PCN([ [fdim, *convlayers[0]], *convlayers[1:] ], output_size, self.dropout) 

        # Define the optimizer
        self.lr = lr
        self.optimizer = getattr(torch.optim, opt)(self.net.parameters(), self.lr)
        # self.optimizer = getattr(torch.optim, opt)(itertools.chain(self.net.parameters(), self.embed.parameters()), lr)

        # Training samples
        self.train_pos = train_pos
        self.val_pos = val_pos
        self.idx = idx
        self.pos = torch.tensor(list(zip(*pos))) #pos = tuple(zip(pos[0],pos[1]))
        self.neg = torch.tensor(list(zip(*neg)))
        
        self.input1 = torch.cat((self.pos[0], self.neg[0]))
        self.input2 = torch.cat((self.pos[1], self.neg[1]))
        self.target = torch.cat((torch.ones(len(self.pos[0])), torch.ones(len(self.neg[0]))*-1)).to(self.device)
        #self.evalpos = pos

        # Loss function
        self.select_loss = select_loss
        self.margin = loss_margin
        self.loss_values = []
        #print('Non-trained loss =',self.lossfunc(self.embed.weight, self.margin).item()])
        
    def lossfunc(self, newemb, margin):
        return F.cosine_embedding_loss(newemb[self.input1], newemb[self.input2], self.target, margin)
    
    """
    # Loss funciton definitions
    def lossfunc(self, newemb, margin):
        if self.select_loss == 'cosine':
            return F.cosine_embedding_loss(newemb[self.input1], newemb[self.input2], self.target, self.margin)
        elif self.select_loss == 'pinsage':
            p = torch.mul(newemb[self.pos[0]], newemb[self.pos[1]]).mean(1).mean()
            n = torch.mul(newemb[self.neg[0]], newemb[self.neg[1]]).mean(1).mean()
            return F.relu(n - p + margin)
    """

    def train(self, epochs, intervals):
      train_eval, val_eval, embs = [], [], []
      [loss_interval, eval_interval, emb_interval] = intervals

      for i in range(1,epochs+1):
          newemb = self.net(self.G, self.embed.weight)
          loss = self.lossfunc(newemb, self.margin)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          if i%loss_interval == 0:
            print('Epoch %d | Loss: %.4f' % (i, loss.item()))
          if i%eval_interval == 0:
            print('Grad / Weights =', (list(self.net.parameters())[1].grad / list(self.net.parameters())[1]).mean().item())
            twopipe = pipe.pipeflow(newemb[self.idx], K=500)
            train_eval.append(twopipe.dfmanip(self.train_pos))
            twopipe = pipe.pipeflow(newemb[self.idx], K=500)
            val_eval.append(twopipe.dfmanip(self.val_pos))

          if i%emb_interval == 0:
            embs.append(newemb)            

          self.loss_values.append(loss.item())
          if loss.item() < 0.01: 
            break

      return [newemb.detach(), train_eval, val_eval, self.loss_values, embs]
