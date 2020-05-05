#%%-------------
""" Cleanse myStory 
import re
import emoji #conda install -c conda-forge emoji
class cleanse:
    #cleanse myStory
    r = re.compile(r'([.,/#!?$%^&*;:{}=_`~()-])[.,/#!?$%^&*;:{}=_`~()-]+')

    @classmethod
    def cleanse(cls, text):
        if (text == '') or (text == None): #.isnull()
            text = -1
        else:
            text = text.replace("\n", ". ") #remove breaks
            text = emoji.get_emoji_regexp().sub(u'',text) #remove emojis
            text = cls.r.sub(r'\1', text) #multiple punctuations
            if len(text) < 10: 
                text = -1 #short texts
        return text

"""
#%%--------------------
""" compute delta(u1, u2) """
import numpy as np
import pandas as pd
from scipy.spatial import distance

class deltas:
    # delta(uv1, uv2) = [ cosine_sim_sbert, count(intersection(iam, meetFor)) equality(marital, has children), abs_diff(age, lat, lng) ]
    df = pd.read_hdf("./data/raw/dproc.h5", key='04')
    ohc_b = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
    ohc_c = np.diag(np.ones(5, dtype=int));  ohc_c[-1] = np.zeros(5, dtype=int)
    ohc_cc = np.diag(np.ones(3, dtype=int))

    @classmethod
    def delta(cls, a, b):
        vec = list()

        #a. cosine_sim_sbert
        emb1 = cls.df.loc[a, 'emb']; emb2 = cls.df.loc[b, 'emb']
        if (type(emb1) or type(emb2)) == int: #len(*emb) > 1
            cos_dist = 0.5
        else:
            cos_dist = 0.5 #round(distance.cdist(emb1, emb2, 'cosine')[0][0], 3)
        vec.append(cos_dist)

        #b. count(intersection(iam, meetFor)) #cls.ohc_b
        iAm = len((cls.df.loc[a, 'iAm']).intersection(cls.df.loc[b, 'iAm']))
        meetFor = len((cls.df.loc[a, 'meetFor']).intersection(cls.df.loc[b, 'meetFor']))
        vec.extend(cls.ohc_b[iAm] + cls.ohc_b[meetFor])
        
        #c. xor(marital, children) #ohc_c, ohc_cc
        ma = int(cls.df.loc[a, 'marital']); mb = int(cls.df.loc[b, 'marital'])
        marital = cls.ohc_c[ma] if (ma == mb) else cls.ohc_c[-1]
        vec.extend(marital.tolist())

        ca = int(cls.df.loc[a, 'children']); cb = int(cls.df.loc[b, 'children'])
        if (ca or cb) == 2:
            children = cls.ohc_cc[cb] if cb!=2 else cls.ohc_cc[ca] if ca!=2 else [0,1,1]
        else: 
            children = cls.ohc_cc[ca] if (ca == cb) else [0,0,0]
        vec.extend(children)

        #d. abs(age, lat, lng)
        age = abs(cls.df.loc[a, 'age'] - cls.df.loc[b, 'age'])/10
        lat = abs(cls.df.loc[a, 'lat'] - cls.df.loc[b, 'lat'])*10
        lng = abs(cls.df.loc[a, 'lng'] - cls.df.loc[b, 'lng'])*10
        vec.extend([age, round(lat, 3), round(lng, 3)])
        return vec

#%% -----------------------------------------------
""" Classification model """
# Build and train a DNN classifier model using the delta vectors.
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump, load

class cmodel:
    #Open dataset, train the model and save
    model = load('./data/model/gbmodel.joblib')
    #def __init__(self, num=0):
    #    self.model = GradientBoostingClassifier(n_estimators=10) if num == 1 \
    #        else load('./data/model/gbmodel.joblib')
    
    def train(self):
        df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')
        trainX, _, trainY, _ = train_test_split(list(df_links['in']), list(df_links['out']), \
                 test_size=0.2, random_state=7)
        #trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.25, random_state=7)
        
        print('-> Training begins with sample sizes: link =', \
            sum(df_links['out']==1), 'No-link =', sum(df_links['out']==0))
        self.model.fit(trainX, trainY)
        dump(self.model, './data/model/gbmodel.joblib')
        print('-> Training has been successfully completed and saved as gbmodel.joblib')
    
    @classmethod
    def modelpred(cls, d):
        return cls.model.predict_proba(d)

#02. DNN MLP

#%%-----------------------
import numpy as np
class recs(cmodel, deltas):
    #def __init__(self):
    #    super(recs, self).__init__(0)

    #@classmethod
    def rec(self, user, list_users):
        # list_users -> delta(a,b) -> c-model probs
        #a. list_users -> delta(a,b)
        d = [deltas.delta(user, i) for i in list_users]
        #b. delta(a,b) -> c-model probs
        probs = cmodel.modelpred(d)[:,0]
        #c. c-model probs -> ranked user
        recs = [list_users[i] for i in np.argsort(probs)]
        return recs


#%%------------


#%%----------------------
"""
df = pd.read_hdf("./data/raw/dproc.h5", key='06')
df.to_hdf("./data/raw/dproc.h5", key='07')
df = df.drop(columns = ['pos', 'neg', 'rest'])
df['recs'] = df.index

import func.op as op
import pandas as pd
df = pd.read_hdf("./data/raw/dproc.h5", key='07')

from importlib import reload
reload(op)
recsys = op.recs()
df['recs'].apply(lambda x: recsys.rec(x, df.loc[x, 'filtered']))

"""
"""
import random

class one:
    n = 5
    @classmethod
    def first(cls, s):
        cls n
        return random.randint(s, cls.n)

class two:
    n = 10
    @classmethod
    def second(cls, s):
        return random.randint(s+5, cls.n)
"""

#%%
#import os

# %%
