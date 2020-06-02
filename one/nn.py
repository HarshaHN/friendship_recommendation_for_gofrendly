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
# from importlib import reload; reload(pinconv)

#%%------------------
#Define the neural network
class neuralnet(nn.Module):
    def __init__(self, in_feats, out_feats, K=2):
        super().__init__()
        # kernels
        self.seq = nn.ModuleList([
            nn.Linear(in_feats, in_feats) for k in range(K) ])
        self.G = nn.Linear(in_feats, out_feats)
        # initializations
        self.seq.apply(self._init_weights)
        self._init_weights(self.G)

    def forward(self, x):
        for i,l in enumerate(self.seq): x = l(x)
        x = F.relu(self.G(x))
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.0)

#%%---------------------------
class net():
   
    def __init__(self, features, pos, neg, outdim, layers, opt, lr, dropout, coslossmargin=0):
        
        # Features
        self.embed = nn.Embedding.from_pretrained(features)
        users, featdim = features.shape

        # Define the model
        self.net = neuralnet(featdim, outdim, layers)

        # Define the optimizer
        self.lr = lr
        self.optimizer = getattr(torch.optim, opt)(self.net.parameters(), self.lr)

        # Training samples
        pos = list(zip(*pos)) #pos = tuple(zip(pos[0],pos[1]))
        neg = list(zip(*pos))
        self.pos = torch.tensor(pos)
        self.neg = torch.tensor(neg)

        # Loss function
        self.margin = coslossmargin
        #self.lossfn = lossfn
        self.loss_values = []
        
        if True: #lossfn == 'cosine':
            self.input1 = torch.cat((self.pos[0], self.neg[0]))
            self.input2 = torch.cat((self.pos[1], self.neg[1]))
            self.target = torch.cat((torch.ones(len(self.pos[0])), torch.ones(len(self.neg[0]))*-1))

    def lossfunc(self, newemb, margin):
        return F.cosine_embedding_loss(newemb[self.input1], newemb[self.input2], self.target, margin)
            
    def train(self, epochs, lossth):
        for i in range(epochs):
            newemb = self.net(self.embed.weight)
            loss = self.lossfunc(newemb, self.margin)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i%25 == 1:
                print('Epoch %d | Loss: %.4f' % (i, loss.item()))
            self.loss_values.append(loss.item())
            if loss.item() < lossth: 
                break

        plt.plot(self.loss_values)
        #list(net.parameters())[0].grad
    
    def eval(pos):
        pass


#%%------------------------
X = torch.cat((numerical_data, categorical_data), 1)
#from nn import net
model = net(features = X,
            pos = trainpos, #402761
            neg = trainneg, #72382
            outdim = 3,
            layers = 2,
            opt = 'Adam',
            lr = 1e-3,
            dropout = 0.2,
            coslossmargin = 0)

#%%----------
model.train(epochs = 10, lossth=0.05)


# %%
emb = model.net(X)