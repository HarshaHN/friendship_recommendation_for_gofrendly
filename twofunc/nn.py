"""
Date: 18 May 2020
Goal: 01 Pinsage algo 1 for one node.
Author: Harsha HN harshahn@kth.se
"""
#%% ------------------------
""" 00. import libraries """
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
dgl.load_backend('pytorch')

#%%-----------------------------------------
""" 01. PinsageConv """
class PinSageConv(nn.Module):

    # init: feature size
    def __init__(self, in_features, out_features, hidden_features): # 3,3,3
        super(PinSageConv, self).__init__()
        # feature size, kernels, params init
        # feature size
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        # kernels
        self.Q = nn.Linear(in_features, hidden_features)
        self.W = nn.Linear(in_features + hidden_features, out_features)
        # params init
        nn.init.xavier_uniform_(self.Q.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W.weight,gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, h_node, h_ngbrs, alpha, gamma='mean'):
        # u* & N(u)**, alpha(equal)**, gamma(mean)
        '''
        h_node(in_features): node embeddings (in_features), or a container (for distributed computing)
        h_ngbrs(num, in_features): node embeddings of its immediate neighbors
        nodeset: node IDs in this minibatch
        alpha(num_neighbors): weight of each neighbor nodes
        gamma('mean'): symmetric vector function
        return(out_features): new node embeddings.
        '''
        # Gather
        num_ngbrs, featuresize = h_ngbrs.shape
        
        # Line 1
        h_ngbrs_one = F.leaky_relu(self.Q(h_ngbrs)) #(num_ngbrs, hidden_features)
        h_agg = self.safediv( # (hidden_features)
                (alpha * h_ngbrs_one).sum(0), #alpha
                alpha.sum(0)) #gamma
        
        # Line 2
        h_concat = torch.cat([h_node, h_agg], 0) # (in_features + hidden_features)
        h_two = F.leaky_relu(self.W(h_concat)) # (out_features)

        # Line 3
        h_new = self.safediv(h_two, h_two.norm()) # (out_features)
        return h_new
    
    # Utility functions
    @staticmethod
    def safediv(a, b):
        b = torch.where(b == 0, torch.ones_like(b), b)
        return a / b