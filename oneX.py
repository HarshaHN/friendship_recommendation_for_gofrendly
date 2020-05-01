
#%%-----------------------------------
""" User info extraction from sql """
import pandas as pd
import pymysql
import func.sql as opsql

# For 'city = Göteborg', get [ user_id, stories, iam, meetFor, birthday, marital, children, lat, lng]
query = {
    '01' : "SELECT user_id, myStory, iAm, meetFor, birthday, marital, children, lat, lng FROM user_details a\
            INNER JOIN users b ON (a.user_id=b.id) WHERE a.city = \"Stockholm\"",
    }
df = opsql.df_sqlquery(query['01'])
del query 
#df.to_hdf("./data/raw/dproc.h5", key='01')

#%%-----------------------------------
""" Data pre-processing - 01 
01. Data cleanse and imputation"""
df = pd.read_hdf("./data/raw/dproc.h5", key='01')
# [ user_id, myStory, values(iAm, meetFor, marital, has child, age, lat, lng) ]
# from ['user_id', 'myStory', 'iAm', 'meetFor', 'age', 'marital', 'children', 'lat', 'lng']

df.set_index('user_id', inplace = True)

#cleanse myStory
def cleanse(text):
    #try:
    if (text == '') or (text == None): #.isnull()
        text = -1
    else:
        import re
        import emoji #conda install -c conda-forge emoji
        text = text.replace("\n", ". ") #remove breaks
        text = emoji.get_emoji_regexp().sub(u'',text) #remove emojis
        r = re.compile(r'([.,/#!?$%^&*;:{}=_`~()-])[.,/#!?$%^&*;:{}=_`~()-]+')
        text = r.sub(r'\1', text) #multiple punctuations
        if len(text) < 10: 
            text = -1 #short texts
    return text

df['myStory'] = df['myStory'].apply(lambda x: cleanse(x))

#'iAm', 'meetFor' to set()
df['iAm'] = df['iAm'].apply(lambda x: set(x.split(',')) if x != None else set())
df['meetFor'] = df['meetFor'].apply(lambda x: set(x.split(',')) if x != None else set())

#'birthday' to age
df['age'] = df['birthday'].apply(lambda x: int((x.today() - x).days/365))
df.drop(columns='birthday', inplace = True)

# has children, marital
df['marital'].fillna(-1, inplace=True)
df['children'].fillna(-1, inplace=True)

#df.to_hdf("./data/raw/dproc.h5", key='02')



#%% -----------------------------------
""" Data pre-processing - 02 
02. stories translation with GCP """

df = pd.read_hdf("./data/raw/dproc.h5", key='02')

#Dummy fill-up
def trans(text):
    #if text == -1: return -1
    #Corpus with example sentences
    texts = [ 'A man is eating food.',
                'A man is eating a piece of bread.',
                'The girl is carrying a baby.',
                'A man is riding a horse.',
                'A woman is playing violin.',
                'Two men pushed carts through the woods.',
                'A man is riding a white horse on an enclosed ground.',
                'A monkey is playing drums.',
                'A cheetah is running behind its prey.']
    import random
    return texts[random.randint(0,8)]

df['myStory'] = df['myStory'].apply(lambda x: trans(x) if x!=-1 else -1)
#df.to_hdf("./data/raw/dproc.h5", key='03')




#%% -----------------------------------
""" Data pre-processing - 03 
03. Stories to S-BERT emb """

df = pd.read_hdf("./data/raw/dproc.h5", key='03')

from sentence_transformers import SentenceTransformer
sbertmodel = SentenceTransformer('roberta-large-nli-mean-tokens') # Load Sentence model (based on BERT) 

def listup(x):
    listx = list()
    listx.append(x)
    return listx

print("-> S-BERT embedding begins...")
import time
start_time = time.time()
df['emb'] = df['myStory'].apply(lambda x: sbertmodel.encode(listup(x)) if x!=-1 else -1)
print("-> S-BERT embedding finished.", (time.time() - start_time)) #534 sec

#df['emb'] = [user_emb['emb'][0], user_emb['emb'][1]]*7474
#df['emb'] = df['emb'].apply(lambda x: x.reshape(1,-1))
df.drop(columns = 'myStory', inplace = True)

#df.to_hdf("./data/raw/dproc.h5", key='04')
#user vectors = [ user_id, 'iAm', 'meetFor', 'marital', 'children', 'lat', 'lng', 'age', 'emb']

#%% -----------------------------------
""" Data pre-processing - 04 
04. Consolidate all the links """

def links(ids):

    import itertools
    """ Links extraction from the network """
    #positive samples
    mf = pd.read_hdf("./data/raw/cmodel.h5", key='mf')
    af = pd.read_hdf("./data/raw/cmodel.h5", key='af')
    #negative samples
    bf = pd.read_hdf("./data/raw/cmodel.h5", key='bf')
    vnf = pd.read_hdf("./data/raw/cmodel.h5", key='vnf')
    print("01 -- vars loaded!")

    #mf as bsp
    bsp = set(tuple(zip(mf.user_id, mf.friend_id)))
    #af as csp
    af = af.groupby(['activity_id'])['user_id'].apply(list)
    csp = set()
    for a in af.iteritems():
        csp.update(set(itertools.combinations(a[1], 2)))

    #bf as asm
    asm = set(tuple(zip(bf.user_id, bf.blocked_id)))
    #vnf as bsm
    bsm = set(tuple(zip(vnf.user_id, vnf.seen_id)))
    print("02 -- links compiled!")

    del mf, af, bf, vnf

    """ Links subset """
    def sublinks(subids, allids):
        temp = set()
        for a,b in allids:
            if (a in subids) and (b in subids):
                temp.add((a,b))
        return temp

    am = sublinks(ids, asm); bm = sublinks(ids, bsm); m = (am | bm)
    ap = sublinks(ids, bsp); cp = sublinks(ids, csp); p = (ap | cp) - m
    print("03 -- classes created!")

    del bsp, csp, asm, bsm, am, ap, cp, bm

    #Save the vars
    import pickle
    filename = './data/vars/links.pickle'
    with open(filename, 'wb') as f:
        pickle.dump([m, p], f)  
    del f, filename

    return [p, m]

df = pd.read_hdf("./data/raw/dproc.h5", key='04')
ids = list(df.index)
[p, n] = links(ids)

#%% -----------------------------------------
""" Data pre-processing - 05 
05. Compute delta of two users """
# delta(uv1, uv2) = [ cosine_sim_sbert, count(intersection(iam, meetFor)) equality(marital, has children), abs_diff(age, lat, lng) ]

import numpy as np
from scipy.spatial import distance

df = pd.read_hdf("./data/raw/dproc.h5", key='04')
global df, ohc_b, ohc_c, ohc_cc
ohc_b = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
ohc_c = np.diag(np.ones(5, dtype=int)); ohc_c[-1] = np.zeros(5, dtype=int)
ohc_cc = np.diag(np.ones(3, dtype=int))

def delta(a, b):
    vec = list()
    global df, ohc_b, ohc_c, ohc_cc

    #a. cosine_sim_sbert
    emb1 = df.loc[a, 'emb']; emb2 = df.loc[b, 'emb']
    if (type(emb1) or type(emb2)) == int: #len(*emb) > 1
        cos_dist = 0.5
    else:
        cos_dist = 0.5 #round(distance.cdist(emb1, emb2, 'cosine')[0][0], 3)
    vec.append(cos_dist)

    #b. count(intersection(iam, meetFor)) #ohc_b
    iAm = len((df.loc[a, 'iAm']).intersection(df.loc[b, 'iAm']))
    meetFor = len((df.loc[a, 'meetFor']).intersection(df.loc[b, 'meetFor']))
    vec.extend(ohc_b[iAm] + ohc_b[meetFor])
    
    #c. xor(marital, children) #ohc_c, ohc_cc
    ma = int(df.loc[a, 'marital']); mb = int(df.loc[b, 'marital'])
    marital = ohc_c[ma] if (ma == mb) else ohc_c[-1]
    vec.extend(marital.tolist())

    ca = int(df.loc[a, 'children']); cb = int(df.loc[b, 'children'])
    if (ca or cb) == 2:
        children = ohc_cc[cb] if cb!=2 else ohc_cc[ca] if ca!=2 else [0,1,1]
    else: 
        children = ohc_cc[ca] if (ca == cb) else [0,0,0]
    vec.extend(children)

    #d. abs(age, lat, lng)
    age = abs(df.loc[a, 'age'] - df.loc[b, 'age'])/10
    lat = abs(df.loc[a, 'lat'] - df.loc[b, 'lat'])*10
    lng = abs(df.loc[a, 'lng'] - df.loc[b, 'lng'])*10
    vec.extend([age, round(lat, 3), round(lng, 3)])
    return vec

#vec = delta(458, 647)

def getlinks():    
    import pickle
    filename = './data/vars/links.pickle'
    with open(filename, 'rb') as f:
        return pickle.load(f) #[m, p]
    return -1
[neg, pos] = getlinks()

# Links to delta 'in'
df_pos = pd.DataFrame({'links': list(pos)})#.head()
df_neg = pd.DataFrame({'links': list(neg)})#.head()

df_pos['in'] = df_pos['links'].apply(lambda x: delta(*x))
df_pos['out'] = [1]*len(df_pos)
print('--> Finished for positive links')
df_neg['in'] = df_neg['links'].apply(lambda x: delta(*x))
df_neg['out'] = [0]*len(df_neg)
print('--> Finished for negative links')

del ohc_b, ohc_c, ohc_cc

## SMOTE ##
#posX = df_pos['in']; posY = df_pos['out']
#negX = df_neg['in']; negY = df_neg['out']
# X = posX + negX # Y = posY + negY  

df_links = pd.concat([df_pos, df_neg], ignore_index=True)
#X = df_links['in]; Y = df_links['out']

#df_links.to_hdf("./data/raw/dproc.h5", key='05')
del df_pos, df_neg


#%% -----------------------------------------------
""" Classification model """
# Build and train a DNN classifier model using the delta vectors.

from sklearn.model_selection import train_test_split
#df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')
# df_pos = df_links[df_links['out']==1]; df_neg = df_links[df_links['out']==0];
#X = df_links['in']; Y = df_links['out']

trainX, testX, trainY, testY = train_test_split(list(df_links['in']), list(df_links['out']), test_size=0.2, random_state=7)
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.25, random_state=7)
print('-> Training begins with sample sizes: link =', sum(df_links['out']==1), 'No-link =', sum(df_links['out']==0))

#01. Gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=10)
model.fit(trainX, trainY)
from joblib import dump
dump(model, './data/model/gbmodel.joblib')

#02. DNN MLP


#Predictions
predY = model.predict(valX)
probY = model.predict_proba(valX)[:,1]
# df.to_hdf("./data/raw/dproc.h5", key='xx')

#%%--------------------------------------------
""" Recsys for all users """
import networkx as nx
import random
import numpy
import pandas as pd

def getlinks():    
    import pickle
    filename = './data/vars/links.pickle'
    with open(filename, 'rb') as f:
        return pickle.load(f) #[m, p]
    return -1
[neg, pos] = getlinks()
posG = nx.Graph(); negG = nx.Graph()

posG.clear(); negG.clear()
posG.add_edges_from(pos); negG.add_edges_from(neg)
#nx.write_graphml(posG, "./data/viz/posG.graphml")
#nx.write_graphml(negG, "./data/viz/negG.graphml")
del neg, pos

rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04')
ids = list(rawdf.index); rest = set(ids) 
df_recsys = pd.DataFrame({'user_id': ids})


df = df_recsys.head()
df['pos'] = df['user_id'].apply(lambda x: set(posG.neighbors(x)) if x in posG else set())
df['neg'] = df['user_id'].apply(lambda x: set(negG.neighbors(x)) if x in negG else set())

df['rest'] = df['user_id']
df.set_index('user_id', inplace = True)
df['rest'] = df['rest'].apply(lambda x: rest - df.loc[x, 'pos'] - df.loc[x, 'neg'])

# Filter the rest using 'user filter values'
df['filtered'] = df['rest'].apply(lambda x: random.sample(list(x), k=5))
del rawdf, ids, df_recsys, negG, posG, rest

# For df['filtered'], compute delta(a,b) and perform c-model.

#import func.op as op
#global a, b
#a = op.delta(); b = op.model()

def recs(user, list_users):
    recs = list()
    global a, b
    # list_users -> delta(a,b) -> c-model probs
    #a. list_users -> delta(a,b)
    deltas = [lambda x: a.delta(user, i) for i in list_users]
    #b. delta(a,b) -> c-model probs
    probs = b.modelpred(deltas)
    #c. c-model probs -> ranked user
    recs = [list_users[i] for i in numpy.argsort(probs)]
    return recs

# df.to_hdf("./data/raw/dproc.h5", key='06')
# df = pd.read_hdf("./data/raw/dproc.h5", key='06')

df['recs'] = df.index
df['recs'] = df['recs'].apply(lambda x: recs(x, df.loc[x, 'filtered']))

del rest, ids
# df_recsys.to_hdf("./data/raw/dproc.h5", key='06')

#%%--------------------------------------------
""" Evaluation of recsys """

import numpy as np
import func.eval as eval
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#0. Confusion matrix
print(classification_report(valY, predY.tolist(), target_names=['No-Link', 'Link']))
confusion_matrix(valY, predY.tolist())

"""1. Metrics of Relevance"""
#1. auroc
"""a. Area under ROC """
def auroc(true, score):
    from sklearn.metrics import roc_auc_score
    res = round(roc_auc_score(true, score),3)
    print("AUROC has been computed and the value is ", res)
    return res

auroc = auroc(valY, probY)

#2. mar@k and map@k
k = 10; query = 1
[map_k, mar_k] = eval.meanavg(1, valY, predY)

#3. hitrate
q = 14
frds = {}
frds[14] = (15, 16, 17)
rec = (12, 13, 14, 15, 16)
hit = eval.hitrate(frds[q], rec)
mrr = eval.mrr(frds[q], rec)

"""2. Metrics of Serendipity"""
"""3. Metrics of User Hits"""
"""4. Rank aware metric"""

# df.to_hdf("./data/raw/dproc.h5", key='07')


#%%--------------------------------------------
from importlib import reload
reload(op)
recsys = op.recs()
df['recs'].apply(lambda x: recsys.rec(x, df.loc[x, 'filtered']))

#%%--------------------------------------------
""" Benchmark the results """

#%%-------------------------------------------
""" Utility ops"""

def clearvars():
    import sys
    sys.modules[__name__].__dict__.clear()
