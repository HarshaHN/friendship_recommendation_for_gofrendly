
# %%--------------------------
""" import libraries """
import pandas as pd
import networkx as nx
import random

#%-----------------
""" PinSAGE """
class gmodel:
    
    def __init__(self, mod=0):
        self.model = mod
        print('-> Model initialised.')
    
    def knn(self, user, K):
        #res = 0 #self.model.knn(user, K)
        return random.sample(range(5536), K)

#%%--------------------------
""" recsystwo """

class recsystwo(gmodel):

    def __init__(self, num):
        super(recsystwo, self).__init__(0)
        self.num = num
        pass

    def dfmanip(self, dvalfrds, K=10):
        df = pd.DataFrame({'id': range(self.num)}) #len(trainids)
        
        #validation set
        valnx = nx.Graph()
        valnx.add_edges_from(dvalfrds)
        df['val'] = df.index
        df['val'] = df['id'].apply(lambda x: list(valnx.neighbors(x)) if x in valnx else -1)
        
        """ #To filter-out seen users
        tempdf = df['id'] #load the tempdf = df['ids', 'filtered']
        df['filtered'] = df['ids'].apply(lambda user: tempdf.loc[id_idx[user], 'filtered'])  """      

        # Run K-NN and get recs 
        df['recs'] = df['id'].apply(lambda user: self.pinrecs(user, K)) #df.loc[x, 'pool']

        # Eval metrics
        df['hitrate'] = df.index; df['mrr'] = df.index
        df['hitrate'] = df['hitrate'].apply(lambda x: self.hitrate(df.loc[x, 'val'], df.loc[x, 'recs']) if df.loc[x, 'val'] != -1 else 0)
        df['mrr'] = df['mrr'].apply(lambda x: self.mrr(df.loc[x, 'val'], df.loc[x, 'recs']) if df.loc[x, 'hitrate']>0 else 0)

        size = valnx.number_of_nodes()
        self.mrr_avg = round(df['mrr'].sum()/size, 3)
        self.hitrate_avg= round(df['hitrate'].sum()/size, 3)
        
        # df.set_index('user_id', inplace = True)
        return df

    def pinrecs(self, user, K):
        # Get K-NN of user who satisfy user's settings.
        #pinrecs = gmodel.knn(self, user, K) 
        return gmodel.knn(self, user, K) #pinrecs

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
