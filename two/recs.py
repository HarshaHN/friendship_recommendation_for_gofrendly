
#%%--------------------------
""" recsystwo """
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import networkx as nx
import random

class recsys():

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
