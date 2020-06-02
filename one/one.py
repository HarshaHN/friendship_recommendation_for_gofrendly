"""
Date: 2 June 2020
Author: Harsha harshahn@kth.se
Semantic similarity of user profiles, output: [auroc, hitrate, mrr]
"""

#%%-----------------------------------
""" 01. Data Loading or Extraction """
from dproc import dproc
# from importlib import reload; reload(dproc)

Load = 2
if Load == 1:
    #['user_id', 'story', 'iam', 'meetfor', 'birthday', 'marital', 'children', 'lat', 'lng']
    users = pd.read_hdf("../data/one/users.h5", key='01')
elif Load == 2: # Load the links
    import pickle
    with open('../data/one/oneload.pkl', 'rb') as f:
        [_, trainpos, trainneg, valpos] = pickle.load(f) # remove users
    del f
else: # Fresh load the data
    users = pd.read_hdf("../data/one/users.h5", key='01')
    [trainpos, trainneg, valpos] = dproc.getdata()

del Load

#%%-----------------------------------
""" 02. Data pre-processing & Feature Engineering """
import dproc
feat = dproc.preproc(users) # Data pre-processing
# feat = pd.read_hdf("../data/one/users.h5", key='05')
feat = dproc.feature(feat)
# feat = pd.read_hdf("../data/one/feat.h5", key='01')
import pipe
[trainX, trainY, valX, valY] = pipe.pipeflow(feat)

categorical_data = torch.tensor(list(feat.cat), dtype=torch.float32)
numerical_data = torch.tensor(list(feat.num), dtype=torch.float32)

id_idx = {id: n for n,id in enumerate(feat.index)} # dict(enumerate(feat.index))
trainpos = [(id_idx[a], id_idx[b]) for a,b in trainpos ]
trainneg = [(id_idx[a], id_idx[b]) for a,b in trainneg ]
#valpos = [(id_idx[a], id_idx[b]) for a,b in valpos ]

X = torch.cat((numerical_data, categorical_data), 1)
# 1024, 46, 3

#%%-------------------
""" 03. Encoder """

#%%------------------------
X = numerical_data
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
model.train(epochs = 10, lossth=0.05)

#%%---------------------
""" 05. Recommendations and evaluation """
emb = model.net(X).detach().numpy()
pipe = pipeflow(emb, K=500, nntype='cosine')

# df, mrr, hr
teval = pipe.dfmanip(trainpos) 
print(teval[1:])

#%%-------------------------------------------
""" Utility ops"""
import sys
sys.modules[__name__].__dict__.clear()

# %%
import h5py
print([key for key in h5py.File('../data/one/users.h5', 'r').keys()])
