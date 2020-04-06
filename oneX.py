"""
Date: 24 Feb 2020
Author: Harsha harshahn@kth.se
Friendship recommendation engine using deep neural network (Social Network Analysis) as a part of Master thesis
"""

# %%
"""import libraries"""
#def importlibs():
    # Tools
import os
import numpy as np
import pandas as pd
import networkx as nx
import time
import func.op as op
import func.sql as opsql
    # ML libraries

#%%
"""SQL connect and query"""
import pymysql
db = opsql.sqlconnect()
opsql.save_sqlquery(db)

# %%
"""Data load"""
uNodes = op.loadone() #load the dfs
#[uNodes, fLinks, aNodes, aLinks] = op.loadone()

#%%
"""a. My story profile match using BERT """
#a. Lang translation to english {German, Swedish, Norwegian} 
stories = pd.concat([uNodes['user_id'], uNodes['myStory']], axis=1)
stories = op.removenull(stories)
stories.to_hdf("stories.h5", key='stories') #save them

#%%
from googletrans import Translator
t = Translator()
t.translate("mitt namn").tex

# %%
#import func.op as op
stories = pd.read_hdf("stories.h5", key='stories')

start_time = time.time()
substories = stories[:50] # take out a sample
substories['story'] = op.trans(substories) # translate
substories = op.removenull(substories)
del substories['myStory']
print("--- %s seconds ---" % (time.time() - start_time))

#%%
substories.to_hdf("stories.h5", key='substories') #save them
substories = pd.read_hdf("stories.h5", key='substories') #load them

# %%
#Sentence Embeddings using BERT / RoBERTa / XLNet https://pypi.org/project/sentence-transformers/
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens') # Load Sentence model (based on BERT) from URL

stories = list(substories['story'])
user_id = list(substories['user_id'])
"""
# Corpus with example sentences
stories = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.']
user_id = list(range(0,len(stories)))"""

#create embeddings
stories_embeddings = model.encode(stories)

#create query embeddings
user = 1; queries = [stories[user]]
query_embeddings = model.encode(queries)

#Find cosine similarity
import scipy
start_time = time.time()
distances = scipy.spatial.distance.cdist(query_embeddings, stories_embeddings, "cosine")
match = 6
ind = distances.argsort()[0].tolist()[:match]

print('Matches for user_id:', user_id[user] , 'with story: \n', queries[0], '\n')

matches = []
for i in ind:
    if (distances[0][i] < 0.5): 
        matches.append(user_id[i])
        print('Match:', i, ', user_id:', user_id[i], 'cosine_dist:', distances[0][i],', story: \n',  stories[i], '\n')
    else: print('No other semantic matches to be found!')


#Elastic search for scalability
#https://xplordat.com/2019/10/28/semantics-at-scale-bert-elasticsearch/


# %%
"""Data Analysis"""
op.loadone() #load the dfs
#Build the Heterogenous Social Network Graph
#G = nx.Graph()


# %%
#Evaluation
import eval

#auroc
true = np.array([0, 0, 1, 1])
score = np.array([0.1, 0.4, 0.35, 0.8])
auroc = eval.auroc(true, score)

#mar@k and map@k
true = [ 0, 0, 1, 1]
pred = [ 1, 0, 0, 1]
k = 10; query = 1
map_k, mar_k = eval.meanavg(query, true, score)

#hitrate
q = 14
frds = {}
frds[14] = (15, 16, 17)
rec = (12, 13, 14, 15, 16)
hit = eval.hitrate(frds[q], rec)
mrr = eval.mrr(frds[q], rec)


# %%
#User interaction and display


# %%
#Training

# %%
#save and commit

# %%
#main


# %%
# Machine Learning based

# %%
# Deep Learning based
