"""
Date: 6 May 2020
Author: Harsha harshahn@kth.se
Graph Neural Network, output: [hitrate, mrr]
"""
#%%------i
import pandas as pd

#%%----------------------
""" 01. Load the data into DGL """
#import dproc
#from importlib import reload; reload(dproc)
from dproc import dproc

Load = 1
if Load == 1:
    import pickle
    with open('../data/two/check01.pkl', 'rb') as f:
        [G, trainpos, trainneg, valpos, _, _] = pickle.load(f)
else: 
    [G, trainpos, trainneg, valpos, _, _] = dproc.getdata()

#%%----------------------
""" 02. Graph Neural Network """

from gnn import gnet
mod = gnet(G, trainpos, trainneg)
#trainpos = tuple(zip(trainpos[0],trainpos[1]))

# import gnn
# from importlib import reload; reload(gnn)
# mod = gnn.gnet(G, trainpos, trainneg)
# del trainneg

#%%-----------------------
""" 03. Train using GCN """
import dgl
#nodefeat = data.getrawemb()

mod.config( fdim = 20, fsize = 3,
            layers = 5,
            opt = 'Adam',
            lr = 1e-3,
            margin = -1.0, # -1 for opp and 0 for 90.
            loss = 'cosine', # cosine or pinsage
            embflag=False, nodefeat=None )

mod.train(epochs = 100, lossth=0.05)

#%%
# Save or load the variables
nodeemb = mod.G.srcdata['feat'].detach().numpy()

"""
import pickle
with open('../data/two/nodeemb.pkl', 'wb') as f:
    pickle.dump(nodeemb, f)

import pickle
with open('../data/two/gnn.pkl', 'wb') as f:
    pickle.dump(mod, f)
del f
"""

import pickle
with open('../data/two/cosineloss.pkl', 'rb') as f:
    nodeemb = pickle.load(f)

with open('../data/two/check01.pkl', 'rb') as f:
    [G, check, _, valpos, _, _] = pickle.load(f) 
    #model = pickle.load('./data/model/gnn.pkl')

#%%----------------------
""" 04. Recsys and evaluation of data """
from recs import recsys
#from importlib import reload; reload(recs)
rec = recsys(G.number_of_nodes(), nodeemb, K=500, nntype='cosine')

# df, mrr, hr
trainpos = tuple(zip(trainpos[0],trainpos[1]))
teval = rec.dfmanip(trainpos) 
veval = rec.dfmanip(valpos)
#testeval = rec.dfmanip(testpos)
print(teval[1:], veval[1:]) #, testeval[1:])

#%%-----------
import sys
sys.modules[__name__].__dict__.clear()
"""
#valpos = zip(*valpos)
#params = list(mod.net.parameters()) #params[0][0]

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
neigh.fit(nodeemb)
neigh.kneighbors(nodeemb, 10, return_distance=False)[0]

resind = [ 0,  210, 2275, 4678, 1905, 5920, 5337, 1618, 1234, 3785]
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(nodeemb, nodeemb)) #res
print(cosine_distances([nodeemb[0]], [nodeemb[210]]))

from scipy.spatial.distance import cosine
print(cosine([nodeemb[0]], [nodeemb[210]]))

import numpy
numpy.argsort([5,4,3,2,1])
"""