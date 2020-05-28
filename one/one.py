
#%%-----------------------------------
""" Data mining (from sql data) """
import pandas as pd
import pymysql

# For 'city = Göteborg', get [ user_id, stories, iam, meetFor, birthday, marital, children, lat, lng]
queries = {
    # User profile
    '01' : "SELECT user_id, myStory, iAm, meetFor, birthday, marital, children, lat, lng FROM user_details a INNER JOIN users b ON (a.user_id=b.id) WHERE a.city = \"Stockholm\"",
    # User friends settings
    '02' : "" #filters
    }

from func.sql import sqlquery
with sqlquery() as newconn:
    df = newconn.query(queries['01']) 
    #uf = newconn.query(queries['02']) 
del queries 
#df.to_hdf("./data/raw/dproc.h5", key='01')

#%%-----------------------------------
""" Data pre-processing """

#%%-----------------------------------
""" Feature Engineering """

#%% -----------------------------------------------
""" Classification model """
# Build and train a DNN classifier model using the delta vectors.
#df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')
from sklearn.externals.joblib import load, dump
from sklearn.neural_network import MLPClassifier
from importlib import reload
import func.cmodel as cmod
reload(cmod)

models = {
    'mlp' : MLPClassifier(hidden_layer_sizes=(50,), alpha=0.001, max_iter=100), #https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    
    'load' : load('./data/model/mlp.pkl')
    }
cmodel = cmod.cmodels(models['mlp'])
model = cmodel.train()
cmodel.eval()
dump(model, './data/model/mlp.pkl')

#%%--------------------------------------------
""" Recsys for all users 
[auc, hitrate, mrr] = recsysone(model, *(df, links))
"""
import func.cmodel as cmod
from importlib import reload
reload(cmod)

mlp = load('./data/model/mlp.pkl')
recsys = cmod.recsysone(mlp)
df = recsys.dfmanip(filtersize = 20, hsize=10)
[auc, hitrate, mrr] = recsys.eval()

#rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04')
#df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')

#%%--------------------------------------------
""" Benchmark the results 
Save the results with config"""

#%%-------------------------------------------
""" Utility ops"""
def clearvars():
    import sys
    sys.modules[__name__].__dict__.clear()

