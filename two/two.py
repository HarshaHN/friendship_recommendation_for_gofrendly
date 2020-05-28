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

#import twofunc
#from importlib import reload; reload(twofunc)
#mod = twofunc.gnn.gnet(G, trainpos, trainneg)

#%%-----------------------
""" 04. Train using GCN """
mod.config(fdim = 10,
                fsize = 3,
                layers = 2,
                opt = 'Adam',
                lr = 1e-3,
                margin = 2.0)

mod.train(epochs = 50)

#%%
# Save or load the variables
"""
emb = mod.G.srcdata['feat'].detach().numpy()
import pickle
with open('./data/two/emb.pkl', 'wb') as f:
    pickle.dump(emb, f)

import pickle
with open('./data/two/mod.pkl', 'wb') as f:
    pickle.dump(mod, f)
"""


"""
"""
import pickle
with open('./data/two/emb.pkl', 'rb') as f:
    emb = pickle.load(f)

with open('./data/two/check01.pkl', 'rb') as f:
    [G, trainpos, _, valpos, _, _] = pickle.load(f)
"""
"""
#%%----------------------
""" 05. Recsys and evaluation of data """
from twofunc.recs import recstwo
recsys = recstwo(G.number_of_nodes(), emb, K=1000)

"""
from importlib import reload; reload(twofunc)
import twofunc
recsys = twofunc.recs.recstwo(G.number_of_nodes(), emb, K=10)
"""

# model = load('./data/model/gnn.pkl')
#recsys = recstwo(num = G.number_of_nodes(), emb=mod.G.srcdata['feat'])
#[traindf, trainmrr, trainhr] = recsys.dfmanip(tuple(zip(trainpos[0],trainpos[1])), 10)
[valdf, valmrr, valhr] = recsys.dfmanip(valpos) #valpos = zip(*valpos) #tuple(zip(trainpos[0],trainpos[1]))
print(valmrr, valhr)

#%%-----------
import sys
sys.modules[__name__].__dict__.clear()



#params = list(mod.net.parameters()) #params[0][0]
#list(mod.G.srcdata['feat'])[0] #embs[0]