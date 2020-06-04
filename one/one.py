"""
Date: 2 June 2020
Author: Harsha harshahn@kth.se
Semantic similarity of user profiles, output: [auroc, hitrate, mrr]
"""

#%%-----------------------------------
""" 01. Load the feature data """
import dproc
import pandas as pd

# sqlusers = pd.read_hdf("../data/one/sqlusers.h5", key='01')     #['user_id', 'story', 'iam', 'meetfor', 'birthday', 'marital', 'children', 'lat', 'lng']
feat = pd.read_hdf("../data/one/trainfeat.h5", key='03') # ['emb', 'cat', 'num'] #dproc: preproc >> feature
[trainpos, trainneg] = dproc.dproc.loadlinks() #46581, 9986

#%%-----------------------------------
""" 02. Transform to model inputs """
categorical_data = torch.tensor(list(feat.cat), dtype=torch.float32)
numerical_data = torch.tensor(list(feat.num), dtype=torch.float32)

ids = list(feat.index)
id_idx = {id: n for n,id in enumerate(feat.index)} # dict(enumerate(feat.index))
trainpos = [(id_idx[a], id_idx[b]) for a,b in trainpos ]
trainneg = [(id_idx[a], id_idx[b]) for a,b in trainneg ]
#valpos = [(id_idx[a], id_idx[b]) for a,b in valpos ]

X = torch.cat((numerical_data, categorical_data), 1)
# 1024, 46, 3

#%%-------------------
""" 03. Encoder Model """

onemodel = net(inputs = X,
            output_size = 24,
            layers = [48, 36, 30],
            dropout = 0.1,
            lr = 1e-3,
            opt = 'Adam', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
            cosine_lossmargin = 0,
            pos = trainpos, #72382
            neg = trainneg) #402761

#model.lr = 5e-3
emb = onemodel.train(epochs = 200, lossth=0.05)
#emb = model.net(X).detach()

#%%---------------------
""" 04. Recommendations and evaluation """

onepipe = onepipeflow(emb, K=500, nntype='cosine')

# df, hr, mrr
onetraindf, onetrainres = onepipe.dfmanip(trainpos)













#%%=================================================
"""
Date: 6 May 2020
Author: Harsha harshahn@kth.se
Graph Neural Network, output: [hitrate, mrr]
"""

G = dproc.dproc.makedgl(num=len(ids), pos=trainpos)
G.readonly(True)


#%%----------------------
""" 01. Graph Neural Network """

#from nn import gnet

twomodel = gnet(graph = G,
                nodeemb = emb,
                convlayers = [24, 24], # hidden, out
                layers = 2,
                output_size = 24,
                dropout = 0.1,
                lr = 1e-3,
                opt = 'Adam', # Rprop, RMSprop, Adamax, AdamW, Adagrad, Adadelta, SGD, Adam
                select_loss = 'cosine',
                loss_margin = 0,
                pos = trainpos, #72382
                neg = trainneg) #402761

#%%--------------
twomodel.lr = 1e-4
newemb = twomodel.train(epochs = 50, lossth=0.05)
print(newemb)
#list(twomodel.parameters())[0].grad


#%%---------------------
""" 02. Recommendations and evaluation """

twoonepipe = onepipeflow(newemb, K=500, nntype='cosine')

# df, hr, mrr
twotraindf, twotrainres = twoonepipe.dfmanip(trainpos)


#%%-----------
import sys
sys.modules[__name__].__dict__.clear()



