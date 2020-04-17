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
#import networkx as nx
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
[uNodes, fLinks, aNodes, aLinks] = op.loadone()

#%%
"""a. My story profile match using BERT """
#a. Lang translation to english {German, Swedish, Norwegian} 
stories = pd.concat([uNodes['user_id'], uNodes['myStory']], axis=1)
#stories.columns = ['user_id', 'story']
stories = op.removenull(stories)
stories.to_hdf("./data/vars/stories.h5", key='stories') #save them

#%%
from googletrans import Translator
t = Translator()
t.translate("mitt namn").text

# %%
import pandas as pd
import func.op as op
import time
stories = pd.read_hdf("./data/vars/stories.h5", key='stories')
stories.columns = ['user_id', 'myStory']
start_time = time.time()
substories = stories[:400] # take out a sample
substories['story'] = op.trans(substories) # translate
substories = op.removenull(substories)
print("--- %s seconds ---" % (time.time() - start_time))
substories = substories.reset_index(drop=True)
substories.to_hdf("./data/vars/stories.h5", key='substories') #save them
del substories['myStory']

#%%
substories.columns = ['user_id', 'myStory']
substories = op.removenull(substories)
substories.columns = ['user_id', 'story']
substories = substories.reset_index(drop=True)
#substories.to_hdf("./data/vars/stories.h5", key='short')
substories = pd.read_hdf("./data/vars/stories.h5", key='short') #load them

# %%
#Sentence Embeddings using BERT / RoBERTa / XLNet https://pypi.org/project/sentence-transformers/
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('roberta-large-nli-mean-tokens') # Load Sentence model (based on BERT) from URL

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
user_id = list(range(0,len(stories)))
"""

#%%
start_time = time.time()
#create embeddings
stories_embeddings = model.encode(stories)
print("--- %s seconds ---" % (time.time() - start_time)) #534 sec
story_emb = pd.DataFrame({'user_id': user_id, 'emb': stories_embeddings, 'story': stories})
story_emb.to_hdf('./data/vars/one.h5', key='emb')
#stories.h5: stories, substories, short
#one.h5: emb

#%%
#create query embeddings
user = 24; queries = [stories[user]]
query_embeddings = model.encode(queries)

#Find cosine similarity
import scipy
start_time = time.time()
distances = scipy.spatial.distance.cdist(query_embeddings, list(story_emb['emb']), "cosine")
match = 6
ind = distances.argsort()[0].tolist()#[:match]
indx = ind[:6] + ind[-6:]
print('Matches for user_id:', user_id[user] , 'with story: \n', queries[0], '\n')

matches = []; count = 0
for i in indx:
    count += 1
    if count == match+1 :
        print('=============================')
        print('---Dissimilar matches here---')
    if True: #(distances[0][i] < 0.5): 
        matches.append(user_id[i])
        print('Match:', i, ', user_id:', user_id[i], 'cosine_dist:', distances[0][i],', story: \n',  stories[i], '\n')
    else: 
        print('No other semantic matches to be found!')

print("--- %s seconds ---" % (time.time() - start_time))

#Elastic search for scalability
#https://xplordat.com/2019/10/28/semantics-at-scale-bert-elasticsearch/

#========================================================================

#%%
""" Friendship links """
import pandas as pd

fLinks = pd.read_hdf("./data/raw/fLinks.h5", key='fLinks')
fList = fLinks.groupby(['user_id'])['friend_id'].apply(list)
fList.to_hdf('./data/vars/fLinks.h5', key='fList')
#fList = pd.read_hdf("./data/vars/fLinks.h5", key='fList')

def getfriends(friends):
    pairs = []
    for user, frds in friends.items():
        for f in frds:
            if (f, user) not in pairs:
                if f in friends.loc[:]:
                    if user in friends.loc[f]:
                        pairs.append((user, f))
    return pairs

fList = pd.read_hdf('./data/vars/fLinks.h5', key='fList')
pairs = getfriends(fList)
fpairs = pd.Series(pairs)
fpairs.to_hdf('./data/vars/fLinks.h5', key='fpairs')
#fpairs = pd.read_hdf("./data/vars/fLinks.h5", key='fpairs')
#fLinks: fList, fpairs,fLinks

#%%
""" Activity links """
import pandas as pd

aLinks = pd.read_hdf('./data/raw/aLinks.h5', key='aLinks')
aList = aLinks.groupby(['activity_id'])['user_id'].apply(list)
aList.to_hdf('./data/vars/aLinks.h5', key='aList')

#========================================================================
#%%
fpairs = pd.read_hdf("./data/vars/fLinks.h5", key='fpairs')
user_emb = pd.read_hdf('./data/vars/one.h5', key='emb')
ids = list(user_emb['user_id'])
subfp =[]
for a in fpairs.iteritems():
    if (a[1][0] in ids) & (a[1][1] in ids):
        subfp.append(a[1])

del a, fpairs, ids

#Save the vars
import pickle
filename = './data/vars/classvars.pickle'
with open(filename, 'wb') as f:
    pickle.dump([user_emb, subfp], f)

del f, filename

#%% 
""" 
Build, train and benchmark a classifier model 

"""
import pickle
filename = './data/vars/classvars.pickle'
with open(filename, 'rb') as f:
    user_emb, subfp = pickle.load(f)





#%%
def clearvars():
    import sys
    sys.modules[__name__].__dict__.clear()



# %%
"""Data Analysis"""
op.loadone() #load the dfs
#Build the Heterogenous Social Network Graph
#G = nx.Graph()


# %%
#Evaluation
import func.eval as eval

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
