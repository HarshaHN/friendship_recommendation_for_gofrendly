
#%%-----------------------------------
"""
Date: 2 June 2020
Author: Harsha harshahn@kth.se
Friendship recommendations based on user profile representation; output: [hitrate, mrr]
"""

#%%-----------------------------------
""" 01. Load the full user data """
import pandas as pd
df1 = pd.read_hdf("../misc/data/uNodes.h5", key='uNodes')
df1.set_index('user_id', inplace = True)
df1 = df1.drop(columns=['isActive', 'isProfileCompleted', 'lang', 'describeYourself', 'iAmCustom', 'meetForCustom'])

df2 = pd.read_hdf("../data/common/train/location.h5", key='info')
df2.set_index('user_id', inplace = True)
df2 = df2.drop(columns=['city', 'country'])

df = pd.concat([df1,df2], axis=1)
df.index.rename('id', inplace=True)
df.columns = ['story', 'iam', 'meetfor', 'marital', 'kids', 'age', 'lat', 'lng'] #'age', 'marital'
df = df.drop([1])

del df1, df2
df.to_hdf('../data/common/train/all_sqlusers', key='01')


#%%-----------------

#'iam' and 'meetfor' to list
df['iam'] = df['iam'].apply(lambda x: [int(i) for i in x.split(',')] if ((x!=None) and (len(x)>0)) else -1)
df['meetfor'] = df['meetfor'].apply(lambda x: [int(i) for i in x.split(',')] if ((x!=None) and (len(x)>0)) else -1)

#'birthday' to age
df['age'] = df['age'].apply(lambda x: int((x.today() - x).days/365)).clip(18, 100).astype('int32') 

# has children, marital
df['marital'] = df['marital'].fillna('-1').astype('int32') # df.marital.value_counts().keys()
df['kids'] = df['kids'].fillna('-1').astype('int32')       # df.kids.value_counts().keys()

# story
mycleanse = cleanse()
df['story'] = df['story'].apply(lambda x: mycleanse.cleanse(x))
df.to_hdf('../data/one/all_processed_users.h5', key='01')


#%%-----------------------------------
""" 01. Load the feature data """
import pandas as pd
import numpy as np

# ['story', 'iam', 'meetfor', 'marital', 'kids', 'age', 'lat', 'lng']
newdf = pd.read_hdf('../data/one/all_processed_users.h5', key='01')
newdf = newdf.drop(columns = ['story'])

# 01. Numerical data
newdf['num'] = newdf.index
newdf['num'] = newdf['num'].apply(lambda x: [newdf.age[x], newdf.lat[x], newdf.lng[x]])
newdf = newdf.drop(columns = ['age', 'lat', 'lng'])

# 02. Categorical data
def onehotencode(input, dim):
    onehot = np.zeros(dim, dtype=int)
    try:
        if isinstance(input, (int, np.integer)):
            onehot[input] = 1
        else: 
            for ind in input:
                if ind < dim: onehot[ind] = 1
    except:
        print('Error for',input)
    return onehot

newdf['cat'] = newdf.index
newdf['cat'] = newdf['cat'].apply(lambda x: np.concatenate(( onehotencode(newdf.iam[x], 18), onehotencode(newdf.meetfor[x], 18), onehotencode(newdf.marital[x], 5), onehotencode(newdf.kids[x], 4) )))
newdf = newdf.drop(columns = ['iam', 'meetfor', 'marital', 'kids'])

import torch
categorical_data = torch.tensor(list(newdf.cat), dtype=torch.float32)
numerical_data = torch.tensor(list(newdf.num), dtype=torch.float32)
all_X = torch.cat((numerical_data, categorical_data), 1)

del categorical_data, numerical_data

import pickle
with open('../data/one/all_X.pkl', 'wb') as f: pickle.dump(all_X, f)
# with open('../data/one/all_X.pkl', 'rb') as f: all_X = pickle.load(f)


#%%
""" 02. Load the full user network connections """

mf = pd.read_hdf("../data/common/train/all_network.h5", key='mf')
bf = pd.read_hdf("../data/common/train/all_network.h5", key='bf')
pos = set(tuple(zip(mf.user_id, mf.friend_id))) #120,409
neg = set(tuple(zip(bf.user_id, bf.blocked_id))) #13,684
valmf = pd.read_hdf("../data/common/val/network.h5", key='mf')
valpos = set(tuple(zip(valmf.user_id, valmf.friend_id))) #126,443
valpos = valpos - pos

id_idx = {id: n for n,id in enumerate(newdf.index)} # dict(enumerate(newdf.index))
trainpos = [(id_idx[a], id_idx[b]) for a,b in pos]
trainneg = [(id_idx[a], id_idx[b]) for a,b in neg]

def sub(links, subids):
    return set([(a,b) for (a,b) in links if ((a in subids) and (b in subids))])

valpos = [(id_idx[a], id_idx[b]) for a,b in sub(valpos, newdf.index)]

with open('../data/one/all_nw.pkl', 'wb') as f: pickle.dump([trainpos, trainneg], f)
with open('../data/one/all_valpos.pkl', 'wb') as f: pickle.dump(valpos, f)
#with open('../data/one/all_nw.pkl', 'rb') as f: [pos, neg] = pickle.load(f)

del bf, f, id_idx, mf, pos, neg, valmf

#%%---------------------------------
with open('../data/one/all_nw.pkl', 'rb') as f: [trainpos, trainneg] = pickle.load(f)
with open('../data/one/all_valpos.pkl', 'rb') as f: valpos = pickle.load(f)
with open('../data/one/all_X.pkl', 'rb') as f: all_X = pickle.load(f)
newdf = pd.read_hdf('../data/one/all_processed_users', key='01')
