"""
Date: 16 May 2020
Author: Harsha harshahn@kth.se
Data pipeline, input: features, output: [hitrate, mrr]
"""

#%%--------------------
import numpy as np
import pandas as pd
import networkx as nx
import random
import torch
import torch.nn.functional as F

class pipeflow:

    def __init__(self, emb, K):
        self.num = emb.shape[0]
        self.emb = emb
        self.K = K+1

    def dfmanip(self, pos):
        df = pd.DataFrame({'id': range(self.num)}) #num of nodes in graph G
        
        # validation set
        actualG = nx.Graph()
        actualG.add_edges_from(pos)
        df['actual'] = df['id'].apply(lambda x: list(actualG.neighbors(x)) if x in actualG else -1)
         
        # Run K-NN and get pred 
        df['pred'] = df['id'].apply(lambda user: self.krecs(user))

        # Eval metrics
        df['hitrate'] = df['id'].apply(lambda x: self.hitrate(df.actual[x], df.pred[x]) if df.actual[x] != -1 else 0)
        df['mrr'] = df['id'].apply(lambda x: self.mrr(df.actual[x], df.pred[x]) if df.hitrate[x]>0 else 0)

        size = actualG.number_of_nodes()
        self.mrr_avg = round(df['mrr'].sum()/size, 3)*100
        self.hitrate_avg= round(df['hitrate'].sum()/size, 3)*100
        
        # df.set_index('user_id', inplace = True)
        print('Hitrate =', self.hitrate_avg, 'MRR =', self.mrr_avg)
        return [self.hitrate_avg, self.mrr_avg]

    def krecs(self, user):
        output = F.cosine_similarity(self.emb[user][None,:], self.emb);
        res = torch.topk(output, self.K).indices.tolist()
        #res = random.sample(range(self.num), self.K)
        return res

    @staticmethod
    def mrr(actual, pred):
        match = list(set(actual).intersection(set(pred)))
        rank = [1 + pred.index(i) for i in match]
        return 1/min(rank)

    @staticmethod
    def hitrate(actual, pred):
        a = set(actual)
        c = a.intersection(set(pred)) 
        res = len(c)/len(a)
        return res
