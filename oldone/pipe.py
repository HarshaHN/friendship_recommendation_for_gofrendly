"""
Date: 16 May 2020
Author: Harsha harshahn@kth.se
Data pipeline, input: features, output: [hitrate, mrr]
"""

#%%--------------------------
"""  """

class pipeflow:

    @staticmethod
    def compdelta():
        # 05. Compute delta of two users

"""
        #05. Compute delta for trainpos and trainneg
        # Links to delta 'in'
        df_pos = pd.DataFrame({'links': list(pos)})#.head()
        df_neg = pd.DataFrame({'links': list(neg)})#.head()

        df_pos['in'] = df_pos['links'].apply(lambda x: delta(*x))
        df_pos['out'] = [1]*len(df_pos)
        print('--> Finished for positive links')
        df_neg['in'] = df_neg['links'].apply(lambda x: delta(*x))
        df_neg['out'] = [0]*len(df_neg)
        print('--> Finished for negative links')

        ## SMOTE ##
        #posX = df_pos['in']; posY = df_pos['out']
        #negX = df_neg['in']; negY = df_neg['out']
        # X = posX + negX # Y = posY + negY  

        df_links = pd.concat([df_pos, df_neg], ignore_index=True) #del df_pos, df_neg
        #X = df_links['in]; Y = df_links['out']
        #df_links.to_hdf("./data/raw/dproc.h5", key='05')
"""

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

        # a. cosine_sim_sbert
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



from sklearn.neighbors import NearestNeighbors
import pandas as pd
import networkx as nx
import random

class pipeflow:

    def __init__(self, num, emb, K, nntype='cosine'):
        self.num = num
        self.emb = emb
        self.K = K
        NN = {
            'knn' : NearestNeighbors(n_neighbors=self.K),
            'cosine': NearestNeighbors(n_neighbors=self.K, algorithm='brute', metric='cosine')
            }
        self.neigh = NN[nntype]
        self.neigh.fit(emb)
        print('-> Model initialised.')


    def dfmanip(self, pos):
        df = pd.DataFrame({'id': range(self.num)}) #num of nodes in graph G
        
        #validation set
        actualG = nx.Graph()
        # pos = tuple(zip(pos[0],pos[1]))
        actualG.add_edges_from(pos)
        df['actual'] = df.index
        df['actual'] = df['id'].apply(lambda x: list(actualG.neighbors(x)) if x in actualG else -1)
        
        """ #To filter-out seen users
        tempdf = df['id'] #load the tempdf = df['ids', 'filtered']
        df['filtered'] = df['ids'].apply(lambda user: tempdf.loc[id_idx[user], 'filtered'])  """      

        # Run K-NN and get recs 
        df['recs'] = df['id'].apply(lambda user: self.pinrecs(user)) #df.loc[x, 'pool']

        # Eval metrics
        df['hitrate'] = df.index; df['mrr'] = df.index
        df['hitrate'] = df['hitrate'].apply(lambda x: self.hitrate(df.loc[x, 'actual'], df.loc[x, 'recs']) if df.loc[x, 'actual'] != -1 else 0)
        df['mrr'] = df['mrr'].apply(lambda x: self.mrr(df.loc[x, 'actual'], df.loc[x, 'recs']) if df.loc[x, 'hitrate']>0 else 0)

        size = actualG.number_of_nodes()
        self.mrr_avg = round(df['mrr'].sum()/size, 3)
        self.hitrate_avg= round(df['hitrate'].sum()/size, 3)
        
        # df.set_index('user_id', inplace = True)
        return [df, self.mrr_avg, self.hitrate_avg]

    def pinrecs(self, user):
        res = self.neigh.kneighbors([self.emb[user]], self.K, return_distance=False)[0]
        # res = random.sample(range(self.num), self.K)
        return list(res[1:])

    @staticmethod
    def mrr(val, recs):
        match = list(set(val).intersection(set(recs)))
        rank = [recs.index(i) for i in match]
        return 1/(min(rank)+1)

    @staticmethod
    def hitrate(val, recs):
        a = set(val)
        c = a.intersection(set(recs)) 
        return len(c)/len(a)

    def eval(self):
        return [self.hitrate_avg, self.mrr_avg] #self.auc
    
