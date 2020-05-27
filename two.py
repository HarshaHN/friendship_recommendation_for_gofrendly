"""
Date: 6 May 2020
Author: Harsha harshahn@kth.se
Graph Neural Network, output: [hitrate, mrr]
"""

#%%----------------------
""" 01. Load the data into DGL """

import twofunc.data as op
from importlib import reload; reload(op)

Load = 1
if Load == 1:
    import pickle
    with open('./data/two/check01.pkl', 'rb') as f:
        [G, trainpos, trainneg, valpos, id_indx, indx_id] = pickle.load(f); del G.ndata['id']
else: [G, trainpos, trainneg, valpos, id_indx, indx_id] = op.createG() #(save=True)
del Load, f, id_indx, indx_id

#%%----------------------
""" 02. Assign features to nodes """

import dgl
#G.ndata['feat'] = op.getrawemb(G.number_of_nodes(), fdim=5, features=False)


#%%----------------------
""" 03. Graph Neural Network """

from twofunc.gnn import gnet
mod = gnet(G, trainpos, trainneg)

import twofunc
from importlib import reload; reload(twofunc)
mod = twofunc.gnn.gnet(G, trainpos, trainneg)

#%%
mod.config(fdim = 6,
                fsize = 3,
                layers = 2,
                opt = 'Adam',
                lr = 1e-5,
                margin = 1.0)

#%%-----------------------
""" 04. Train using GCN """
mod.train(epochs = 2)

#%%----------------------
""" 05. Recsys and evaluation of data """
import twofunc.recs as gnn
# from importlib import reload; reload(gnn)

# model = load('./data/model/gnn.pkl')
recsys = gnn.recsystwo(len(trainids))
recsdf = recsys.dfmanip(trainfrds, 10)
recsdf = recsys.dfmanip(dvalfrds, 10)
[hitrate, mrr] = recsys.eval()





#%%-----------
import sys
sys.modules[__name__].__dict__.clear()


"""
To-Do
a. Train G with mf and bf
b. implement KNN and eval on validation dataset
c. node embeddings for isolated nodes???
d. feature scaling and inclusion
e. integrate all and code for experiments.
"""