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
   
    def __init__(self, in_feats, hidden_feats, out_feats):
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

    def forward(self, graph, feat):
        """Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature (Node embeddings)

        Returns
        -------
        torch.Tensor
            The output feature (Node embeddings)
        """
        graph = graph.local_var()

        graph.srcdata['h'] = F.relu(self.Q(feat))
        
        def mfunc(edges):
            return {'m':edges.src['h'], 'a':edges.data['w']}

        def rfunc(nodes):
            m = nodes.mailbox['m']
            a = nodes.mailbox['a'] 
            res = th.mul(a[:,:,None], m).sum(1)
            res = res / a.sum(1)[:, None]
            return {'h': res}

        graph.update_all(mfunc, rfunc)

        rst = graph.dstdata['h']
        rst = th.cat([feat, rst], 1)
        rst = F.relu(self.W(rst))
        
        degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
        norm = 1.0 / degs
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = th.reshape(norm, shp)
        rst = rst * norm

        denom = rst.norm(dim=1, keepdim=True)
        if any(denom) != 0:
            rst = rst / denom

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

