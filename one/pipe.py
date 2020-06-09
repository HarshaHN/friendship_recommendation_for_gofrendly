"""
Date: 16 May 2020
Author: Harsha harshahn@kth.se
Data pipeline, input: features, output: [hitrate, mrr]
"""

#%%--------------------
import numpy as np
import pandas as pd
from scipy.spatial import distance

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import networkx as nx
import random

class pipeflow:

    def __init__(self, emb, K, nntype='cosine'):
        self.num, fdim = emb.shape
        self.emb = emb.numpy()
        self.K = K
        NN = {
            'knn' : NearestNeighbors(n_neighbors=self.K),
            'cosine': NearestNeighbors(n_neighbors=self.K, algorithm='brute', metric='cosine')
            }
        self.neigh = NN[nntype]
        self.neigh.fit(emb)
        print('-> Nearest neighbor model initialised.')


    def dfmanip(self, pos):
        df = pd.DataFrame({'id': range(self.num)}) #num of nodes in graph G
        
        #validation set
        actualG = nx.Graph()
        actualG.add_edges_from(pos)
        df['actual'] = df.index
        df['actual'] = df['id'].apply(lambda x: list(actualG.neighbors(x)) if x in actualG else -1)
        
        """ #To filter-out seen users
        tempdf = df['id'] #load the tempdf = df['ids', 'filtered']
        df['filtered'] = df['ids'].apply(lambda user: tempdf.loc[id_idx[user], 'filtered'])  """      

        # Run K-NN and get recs 
        df['recs'] = df['id'].apply(lambda user: self.krecs(user))

        # Eval metrics
        df['hitrate'] = df['id'].apply(lambda x: self.hitrate(df.loc[x, 'actual'], df.loc[x, 'recs']) if df.loc[x, 'actual'] != -1 else 0)
        df['mrr'] = df['id'].apply(lambda x: self.mrr(df.loc[x, 'actual'], df.loc[x, 'recs']) if df.loc[x, 'hitrate']>0 else 0)

        size = actualG.number_of_nodes()
        self.mrr_avg = round(df['mrr'].sum()/size, 3)
        self.hitrate_avg= round(df['hitrate'].sum()/size, 3)
        
        # df.set_index('user_id', inplace = True)
        print('Hitrate =', self.hitrate_avg*100, 'MRR =', self.mrr_avg*100)
        return [df, [self.hitrate_avg, self.mrr_avg]]

    def krecs(self, user):
        res = list(self.neigh.kneighbors([self.emb[user]], self.K, return_distance=False)[0][1:])
        # res = random.sample(range(self.num), self.K)
        return res

    @staticmethod
    def mrr(actual, recs):
        match = list(set(actual).intersection(set(recs)))
        rank = [recs.index(i) for i in match]
        return 1/(min(rank)+1)

    @staticmethod
    def hitrate(actual, recs):
        a = set(actual)
        c = a.intersection(set(recs)) 
        return len(c)/len(a)

    def eval(self):
        pass

#%%

