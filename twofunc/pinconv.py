"""Torch modules for pinsage graph convolutions(PGCN)."""

import torch as th
from torch import nn
from torch.nn import init
import dgl.function as fn
import torch.nn.functional as F
#from ....base import DGLError

class PinConv(nn.Module):
   
 def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 norm='right'):
        super(PinConv, self).__init__()
        self._in_feats = in_feats
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats
        self._norm = norm

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

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()

        graph.srcdata['h'] = F.relu(self.Q(feat))
        graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
        rst = graph.dstdata['h']
        rst = th.cat([feat, rst], 1)
        rst = F.relu(self.W(rst))

        degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
        norm = 1.0 / degs
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = th.reshape(norm, shp)
        rst = rst * norm

        rst = rst / rst.norm(dim=1, keepdim=True)
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