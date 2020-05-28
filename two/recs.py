
#%%--------------------------
""" recsystwo """
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import networkx as nx
import random

class recstwo():

    def __init__(self, num, emb, K):
        #super(recstwo, self).__init__(num, emb, K, mod=0)
        self.num = num
        self.emb = emb
        self.K = K
        self.neigh = NearestNeighbors(n_neighbors=self.K)
        self.neigh.fit(emb)
        print('-> Model initialised.')


    def dfmanip(self, pos):
        df = pd.DataFrame({'id': range(self.num)}) #num of nodes in graph G
        
        #validation set
        valnx = nx.Graph()
        # pos = tuple(zip(pos[0],pos[1]))
        valnx.add_edges_from(pos)
        df['val'] = df.index
        df['val'] = df['id'].apply(lambda x: list(valnx.neighbors(x)) if x in valnx else -1)
        
        """ #To filter-out seen users
        tempdf = df['id'] #load the tempdf = df['ids', 'filtered']
        df['filtered'] = df['ids'].apply(lambda user: tempdf.loc[id_idx[user], 'filtered'])  """      

        # Run K-NN and get recs 
        df['recs'] = df['id'].apply(lambda user: self.pinrecs(user)) #df.loc[x, 'pool']

        # Eval metrics
        df['hitrate'] = df.index; df['mrr'] = df.index
        df['hitrate'] = df['hitrate'].apply(lambda x: self.hitrate(df.loc[x, 'val'], df.loc[x, 'recs']) if df.loc[x, 'val'] != -1 else 0)
        df['mrr'] = df['mrr'].apply(lambda x: self.mrr(df.loc[x, 'val'], df.loc[x, 'recs']) if df.loc[x, 'hitrate']>0 else 0)

        size = valnx.number_of_nodes()
        self.mrr_avg = round(df['mrr'].sum()/size, 3)
        self.hitrate_avg= round(df['hitrate'].sum()/size, 3)
        
        # df.set_index('user_id', inplace = True)
        return [df, self.mrr_avg, self.hitrate_avg]

    def pinrecs(self, user):
        # Get K-NN of user who satisfy user's settings.
        res = self.neigh.kneighbors(self.emb[user][None,:], self.K, return_distance=False)
        # res = random.sample(range(self.num), self.K)
        #pinrecs = gmodel.knn(self, user) 
        return list(res[0])

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


#%%-----------------
"""
class gmodel:
    
    def __init__(self, num, emb, K, mod=0):
        self.num = num
        self.emb = emb
        from sklearn.neighbors import NearestNeighbors
        self.neigh = NearestNeighbors(n_neighbors=K)
        self.neigh.fit(emb)
        print('-> Model initialised.')
    
    def knn(self, user, K):
        self.neigh.kneighbors(X, K, return_distance=False)
                #res = 0 #self.model.knn(user, K)
        return random.sample(range(self.num), K)
"""