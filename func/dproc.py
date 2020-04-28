
#%%
import pandas as pd
import pymysql
import func.sql as opsql

# For 'city = Göteborg', get [ user_id, stories, iam, meetFor, birthday, marital, children, lat, lng]
# df = df_sqlquery(query)
query = {
    '01' : "SELECT user_id, myStory, iAm, meetFor, birthday, marital, children, lat, lng FROM user_details a\
            INNER JOIN users b ON (a.user_id=b.id) WHERE a.city = \"Stockholm\"",
    }
df = opsql.df_sqlquery(query['01'])
del query 
df.to_hdf("./data/raw/dproc.h5", key='01')
#------------------------------------------------------------------------------------

#%%
df = pd.read_hdf("./data/raw/dproc.h5", key='01')
# [ user_id, myStory, values(iAm, meetFor, marital, has child, age, lat, lng) in range(0,1) ]
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
#df['children'] = df['children'].apply(lambda x: 1 if (x>=0) else -1)

df.to_hdf("./data/raw/dproc.h5", key='02')
#--------------------------------------------
#%% stories translation with GCP
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
#-------------------------------------------------------------------

#%% Stories to S-BERT emb
df = pd.read_hdf("./data/raw/dproc.h5", key='03')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('roberta-large-nli-mean-tokens') # Load Sentence model (based on BERT) 

def listup(x):
    listx = list()
    listx.append(x)
    return listx

print("-> S-BERT embedding begins...")
import time
start_time = time.time()
df['emb'] = df['myStory'].apply(lambda x: model.encode(listup(x)) if x!=-1 else -1)
print("-> S-BERT embedding finished.", (time.time() - start_time)) #534 sec

#df['emb'] = [user_emb['emb'][0], user_emb['emb'][1]]*7474
#df['emb'] = df['emb'].apply(lambda x: x.reshape(1,-1))
df.drop(columns = 'myStory', inplace = True)

df.to_hdf("./data/raw/dproc.h5", key='04')
#user vectors = [ user_id, sbert_emb, iam, meetFor, marital, children, age, lat, lng ]
#test.to_hdf("./data/raw/dproc.h5", key='test')
#----------------------------------------------------------------------------

#%% 
# df = pd.read_hdf("./data/raw/dproc.h5", key='04')
# delta(uv1, uv2) = [ cosine_sim_sbert, count(intersection(iam, meetFor)/3) xor(marital, has children), abs_diff(age, lat, lng) ]
# one-hot encode

import numpy as np
from scipy.spatial import distance

global df, ohc_b, ohc_c, ohc_cc
ohc_b = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
ohc_c = np.diag(np.ones(5)); ohc_c[-1] = np.zeros(5)
ohc_cc = np.diag(np.ones(3))

def delta(a, b):
    vec = list()
    global df, ohc_b, ohc_c, ohc_cc

    #a. cosine_sim_sbert
    emb1 = df.loc[a, 'emb']; emb2 = df.loc[b, 'emb']
    if (type(emb1) or type(emb2)) == int: #len(*emb) > 1
        print('1: ', type(emb1),'2: ', type(emb2))
        cos_dist = [0.5]
    else:
        cos_dist = distance.cdist(df.loc[a, 'emb'], df.loc[b, 'emb'], 'cosine').tolist()
    vec.append(*cos_dist)

    #b. count(intersection(iam, meetFor))
    #ohc_b = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
    iAm = len((df.loc[a, 'iAm']).intersection(df.loc[b, 'iAm']))
    meetFor = len((df.loc[a, 'meetFor']).intersection(df.loc[b, 'meetFor']))
    vec += [ohc_b[iAm]] + [ohc_b[meetFor]]  
    
    #c. xor(marital, children)
    #ohc_c = np.diag(np.ones(5)); ohc_c[-1] = np.zeros(5)
    ma = int(df.loc[a, 'marital']); mb = int(df.loc[b, 'marital'])
    marital = ohc_c[ma] if (ma == mb) else ohc_c[-1]

    #ohc_cc = np.diag(np.ones(3))
    ca = int(df.loc[a, 'children']); cb = int(df.loc[b, 'children'])
    if (ca or cb) == 2:
        children = ohc_cc[cb] if cb!=2 else ohc_cc[ca] if ca!=2 else [0,1,1]
    else: 
        children = ohc_cc[ca] if (ca == cb) else [0,0,0]
    vec += [marital.tolist()] + [children]

    #d. eucld(age, lat, lng)
    age = abs(df.loc[a, 'age'] - df.loc[b, 'age'])#/10
    lat = abs(df.loc[a, 'lat'] - df.loc[b, 'lat'])
    lng = abs(df.loc[a, 'lng'] - df.loc[b, 'lng'])
    vec += [[age, lat, lng]]
    
    return vec

vec = delta(458, 647)

del ohc_b, ohc_c, ohc_cc

#%%
#Consolidate all the links
def links(ids):

    import itertools
    """ Links extraction from the network """
    #positive samples
    mf = pd.read_hdf("./data/raw/cmodel.h5", key='mf')
    af = pd.read_hdf("./data/raw/cmodel.h5", key='af')
    #negative samples
    bf = pd.read_hdf("./data/raw/cmodel.h5", key='bf')
    vnf = pd.read_hdf("./data/raw/cmodel.h5", key='vnf')

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

    del mf, af, bf, vnf

    """ Links subset """
    def sublinks(subids, allids):
        temp = set()
        for a,b in allids:
            if (a in subids) and (b in subids):
                temp.add((a,b))
        return temp

    m = (sublinks(ids, asm) | sublinks(ids, bsm))
    p = (sublinks(ids, bsp) | sublinks(ids, csp)) - m

    del bsp, csp, asm, bsm

    return [p, m]

ids = list(df.index)
[p, n] = links(ids)
# posX = list(); posY = list();
# for (a,b) in p:
#   posX.append(delta(a,b))
#   posY.append(1) #perhaps a one-hot code label
# for (a,b) in n:
#   negX.append(delta(a,b))
#   negY.append(0) #perhaps a one-hot code label
# 
# SMOTE if needed
# X = posX + negX
# Y = posY + negY  

# df.to_hdf("./data/raw/dproc.h5", key='05')
#--------------------------------------------
# %%


# df.to_hdf("./data/raw/dproc.h5", key='06')
#--------------------------------------------

#%%

""" Classification model """

# Build and train a DNN classifier model using the delta vectors.

"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

model_GB = GradientBoostingClassifier(n_estimators=50)
model_GB.fit(xf_train, y_train)
y_pred = model_GB.predict(xf_test)
target_names = ['Link', 'No-Link']
print(classification_report(y_test, y_pred.tolist(), target_names=target_names))
confusion_matrix(y_test, y_pred.tolist())
"""

# df.to_hdf("./data/raw/dproc.h5", key='07')
#--------------------------------------------

#%%
# Benchmark the results
"""
import func.eval as eval
import numpy as np

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

# df.to_hdf("./data/raw/dproc.h5", key='08')
#--------------------------------------------
"""