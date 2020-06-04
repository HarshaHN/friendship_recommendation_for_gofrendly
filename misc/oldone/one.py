"""
Date: 16 March 2020
Author: Harsha harshahn@kth.se
Semantic similarity of user profiles, output: [auroc, hitrate, mrr]
"""

#%%-----------------------------------
""" 01. Data Loading or Extraction """

import dproc
from importlib import reload; reload(dproc)

Load = 1
if Load == 1:
    users = pd.read_hdf("../data/one/users.h5", key='01')
elif: Load == 2:
    import pickle
    with open('../data/one/oneload.pkl', 'rb') as f:
        [users, trainpos, trainneg, valpos] = pickle.load(f)
    del f, Load
else: 
    [users, trainpos, trainneg, valpos] = dproc.getdata()

#%%-----------------------------------
""" 02. Data pre-processing & Feature Engineering """
import dproc
feat = dproc.preproc(users)
feat = pd.read_hdf("../data/one/users.h5", key='05')

import pipe
[trainX, trainY, valX, valY] = pipe.pipeflow(feat) #compute delta of all trainpos and trainneg


#%% -----------------------------------------------
""" 03. Train a classification model """
# Build and train a DNN classifier model using the delta vectors.
#df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')
from sklearn.externals.joblib import load, dump
from sklearn.neural_network import MLPClassifier
from importlib import reload
import cmodel as cmod
reload(cmod)

models = {
    'mlp' : MLPClassifier(hidden_layer_sizes=(50,), alpha=0.001, max_iter=100), #https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    
    'load' : load('./data/model/mlp.pkl')
    }
cmodel = cmod.cmodels(models['mlp'])
model = cmodel.train([trainX, trainY])
cmodel.eval()
dump(model, './data/model/mlp.pkl')

#%%--------------------------------------------
""" 04. Friendship recommendations """
# [auc, hitrate, mrr] = recsysone(model, *(df, links))

import cmodel as cmod
from importlib import reload
reload(cmod)

mlp = load('./data/model/mlp.pkl')
recsys = cmod.recsysone(mlp)
df = recsys.dfmanip(filtersize = 20, hsize=10)
teval = predict([trainX, trainY]) #, valX, valY
veval = predict([valX, valY]) #, 
print(teval, veval)
[auc, hitrate, mrr] = recsys.eval()

#rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04')
#df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')





#%%-------------------------------------------
""" Utility ops"""
import sys
sys.modules[__name__].__dict__.clear()


# %%
import h5py
print([key for key in h5py.File('../data/one/users.h5', 'r').keys()])
