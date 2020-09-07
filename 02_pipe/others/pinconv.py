"""
Date: 18 May 2020
Goal: 01 Pinsage algo 1 for one node.
Author: Harsha HN harshahn@kth.se
"""

#%% ------------------------
""" 00. import libraries """
import torch as th
from torch import nn
from torch.nn import init
import dgl.function as fn
import torch.nn.functional as F
#dgl.load_backend('pytorch')


""" 01. PinsageConv """
class PinConv(nn.Module):

    def __init__(self, in_feats, hidden_feats, out_feats, dropout=0):
        super(PinConv, self).__init__()
        # feature size
        self._in_feats = in_feats
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats 
        
        # kernels
        self.Q = nn.Linear(in_feats, hidden_feats)
        self.W = nn.Linear(in_feats + hidden_feats, out_feats)
        # params init
        nn.init.xavier_uniform_(self.Q.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.Q.bias, 0)
        nn.init.constant_(self.W.bias, 0)

        # Norm
        self.bnorm_one = nn.BatchNorm1d(hidden_feats)
        self.bnorm_two = nn.BatchNorm1d(out_feats)

        # Dropouts
        self.dropout_one = nn.Dropout(dropout)
        self.dropout_two = nn.Dropout(dropout)

        #device
        #self.device = xm.xla_device() 
        self.device = th.device('cuda')# if th.cuda.is_available() else 'cpu')

    def forward(self, graph, feat):
        graph = graph.local_var()

        graph.srcdata['h'] = self.dropout_one(self.bnorm_two(F.relu(self.Q(feat)))).to(self.device)

        def mfunc(edges):
            return {'m':edges.src['h'], 'a':edges.data['w']}

        def rfunc(nodes):
            m = nodes.mailbox['m'].to(self.device)
            a = nodes.mailbox['a'].to(self.device)
            res = th.mul(a[:,:,None], m).sum(1)
            res = res / a.sum(1)[:, None]
            return {'h': res}

        graph.update_all(mfunc, rfunc)

        rst = graph.dstdata['h']
        rst = th.cat([feat, rst], 1)
        
        rst = self.dropout_two(self.bnorm_two(F.relu(self.W(rst))))

        denom = rst.norm(dim=1, keepdim=True)
        if any(denom) != 0:
            rst = rst / denom

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        #summary += ', normalization={_norm}'
        #if '_activation' in self.__dict__:
        #    summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

