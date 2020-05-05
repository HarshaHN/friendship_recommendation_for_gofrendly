

#%%--------------
""" Recsys for all users """
import func.op as op
import pandas as pd
from importlib import reload
reload(op)

rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04')
df = op.links_filt(list(rawdf.index), filtersize = 20, hsize=10)


# df.to_hdf("./data/raw/dproc.h5", key='06')
# df = pd.read_hdf("./data/raw/dproc.h5", key='06')

mlp = load('./data/model/mlp.pkl')
recsys = op.recs(mlp)
df['recs'] = df['recs'].apply(lambda x: recsys.rec(x, df.loc[x, 'filtered']))

import func.lab as lab

# df.to_hdf("./data/raw/dproc.h5", key='07')
# df_recsys.to_hdf("./data/raw/dproc.h5", key='06')


import random
import math
from sklearn.externals.joblib import load, dump
import func.op as op

#1. takeout 20% of pos into 'val'
df['val'] = df['pos'].apply(lambda x: random.sample(list(x), min(math.floor(len(x)*0.25), 5)) if len(x)>3 else None)
df = df.dropna()

#2. 
df['pool'] = df.index
df['pool'] = df['pool'].apply(lambda x: df.loc[x, 'filtered'] + df.loc[x, 'val'])


mlp = load('./data/model/mlp.pkl')
recsys = op.recs(mlp)
df['recs'] = df.index
df['recs'] = df['recs'].apply(lambda x: recsys.rec(x, df.loc[x, 'pool']))
df = df.drop(columns = ['neg', 'recs', 'rest','pool', 'pos'])

df['hitrate'] = df.index
df['hitrate'] = df['hitrate'].apply(lambda x: hitrate(df.loc[x, 'val'], df.loc[x, 'recs'][:10]))

df['mrr'] = df.index
df['mrr'] = df['mrr'].apply(lambda x: mrr(df.loc[x, 'val'], df.loc[x, 'recs']) if df.loc[x,'hitrate']>0 else 0)

mrr = df['mrr'].sum()/len(df)
hitrate = df['hitrate'].sum()/len(df)

def mrr(val, recs):
    rank = [recs.index(i) for i in val]
    return 1/(min(rank)+1)

def hitrate(val, recs):
    a = set(val)
    c = a.intersection(set(recs)) 
    return len(c)/len(a)