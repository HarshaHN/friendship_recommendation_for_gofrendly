#%%--------------------
""" Compute input: delta(u1, u2) """
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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from joblib import dump, load
import time
#import func.eval as eval

class cmodels:
    #Open dataset, train the model and save
    df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')
    auc = 0.0

    def __init__(self, mod):
        self.model = mod
        print('->Model initialised.')

    def train(self):
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(list(self.df_links['in']),\
            list(self.df_links['out']), test_size=0.2, random_state=7)
        self.trainX, self.valX, self.trainY, self.valY = train_test_split(self.trainX, self.trainY, \
            test_size=0.25, random_state=7)
        
        start_time = time.time()
        print('-> Training begins with sample sizes: link =', \
            self.trainY.count(1), ', No-link =', self.trainY.count(0))
        self.model.fit(self.trainX, self.trainY)
        print('-> Training has been successfully completed in sec', (time.time() - start_time))
        return self.model
    
    @staticmethod
    def auroc(true, score):
        res = round(roc_auc_score(true, score),3)
        print("AUROC has been computed and the value is ", res)
        return res

    def predprob(self, x):
        return self.model.predict_proba(x)
        
    def eval(self):
        #00. Sanity test
        predY = self.model.predict(self.valX).tolist()
        print('Classification report: \n',classification_report(self.valY, predY, target_names=['No-Link', 'Link']))
        #print('Confusion matrix: \n' ,confusion_matrix(self.valY, predY, labels = [0, 1]))
        #01. AUROC
        probY = self.model.predict_proba(self.valX)[:,1]
        self.auc = cmodels.auroc(self.valY, probY)
        print('-> AUROC =', self.auc)


#%%-----------------------
""" Recsys for all users """
import numpy as np
import pandas as pd
import func.eval as eval
import networkx as nx
import random
import pickle
import math
from sklearn.externals.joblib import load

class recsysone(cmodels, deltas):
    rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04')
    #df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')

    def __init__(self, mod):
        super(recsysone, self).__init__(mod)

    def dfmanip(self, filtersize = 20, hsize=10):
        """
        filtersize: user settings filter 
        hsize: user size under consideration
        """
        df = self.links_filt(list(self.rawdf.index), hsize=10) #get 'pos' and 'neg'
        # 01. Get recommendation list
        df.set_index('user_id', inplace = True)
        df['rest'] = df.index; df['pool'] = df.index; df['recs'] = df.index
        
        #validation set
        df['val'] = df['pos'].apply(lambda x: random.sample(list(x), min(math.floor(len(x)*0.25), 5)) \
            if len(x)>3 else None)

        df['rest'] = df['rest'].apply(lambda x: set(self.rawdf.index) - df.loc[x, 'pos'] - df.loc[x, 'neg'])
        df['filtered'] = df['rest'].apply(lambda x: random.sample(list(x), filtersize))
        df = df.dropna()
        df['pool'] = df['pool'].apply(lambda x: df.loc[x, 'filtered'] + df.loc[x, 'val'])
        df['recs'] = df['recs'].apply(lambda x: self.rec(x, df.loc[x, 'pool']))
        df = df.drop(columns = ['pos', 'neg', 'rest', 'filtered', 'pool'])

        df['hitrate'] = df.index; df['mrr'] = df.index
        df['hitrate'] = df['hitrate'].apply(lambda x: self.hitrate(df.loc[x, 'val'], df.loc[x, 'recs'][:10]))
        df['mrr'] = df['mrr'].apply(lambda x: self.mrr(df.loc[x, 'val'], df.loc[x, 'recs']) if df.loc[x,'hitrate']>0 else 0)

        self.mrr_avg = df['mrr'].sum()/len(df)
        self.hitrate_avg= df['hitrate'].sum()/len(df)
        return df
    
    def rec(self, user, list_users):
        # list_users -> delta(a,b) -> c-model probs
        #a. list_users -> delta(a,b)
        d = [deltas.delta(user, i) for i in list_users]
        #b. delta(a,b) -> c-model probs
        probs = cmodels.predprob(self, d)[:,0]
        #c. c-model probs -> ranked user
        recs = [list_users[i] for i in np.argsort(probs)]
        return recs

    @staticmethod
    def mrr(val, recs):
        rank = [recs.index(i) for i in val]
        return 1/(min(rank)+1)

    @staticmethod
    def hitrate(val, recs):
        a = set(val)
        c = a.intersection(set(recs)) 
        return len(c)/len(a)

    def eval(self):
        return [self.auc, self.hitrate_avg, self.mrr_avg]

    """ User links -> pos_neg list """
    def links_filt(self, ids, hsize=10):
            filename = './data/vars/links.pickle'
            with open(filename, 'rb') as f:
                [neg, pos] = pickle.load(f)
            posG = nx.Graph(); negG = nx.Graph()
            posG.add_edges_from(pos); negG.add_edges_from(neg)
            #nx.write_graphml(posG, "./data/viz/posG.graphml")
            #nx.write_graphml(negG, "./data/viz/negG.graphml")
            # Pos and Neg
            df = pd.DataFrame({'user_id': ids[:hsize]})
            df['pos'] = df['user_id'].apply(lambda x: set(posG.neighbors(x)) \
                if x in posG else set())
            df['neg'] = df['user_id'].apply(lambda x: set(negG.neighbors(x)) \
                if x in negG else set())
            return df

#%%------------------------------------
"""
import random

class one:
    n = 5
    @classmethod
    def first(cls, s):
        return random.randint(s, cls.n)

class two:
    n = 10
    @classmethod
    def second(cls, s):
        return random.randint(s+5, cls.n)
"""