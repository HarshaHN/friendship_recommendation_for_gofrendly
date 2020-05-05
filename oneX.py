
#%%-----------------------------------
""" User info extraction from sql """
import pandas as pd
import pymysql

# For 'city = Göteborg', get [ user_id, stories, iam, meetFor, birthday, marital, children, lat, lng]
queries = {
    # User profile
    '01' : "SELECT user_id, myStory, iAm, meetFor, birthday, marital, children, lat, lng FROM user_details a INNER JOIN users b ON (a.user_id=b.id) WHERE a.city = \"Stockholm\"",
    # User friends settings
    '02' : ""
    }

"""
from func.sql import sqlquery
with sqlquery() as newconn:
    df = query(queries['01']) 

from func.sql import df_sqlquery
#df = df_sqlquery(queries['01'])
"""
del queries 
#df.to_hdf("./data/raw/dproc.h5", key='01')

#%%-----------------------------------
""" Data pre-processing - 01 
01. Data cleanse and imputation"""
df = pd.read_hdf("./data/raw/dproc.h5", key='01')
# from ['user_id', 'myStory', 'iAm', 'meetFor', 'age', 'marital', 'children', 'lat', 'lng']
# To [ user_id, myStory, values(iAm, meetFor, marital, has child, age, lat, lng) ]

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

"""
mycleanse = cleanse(); mycleanse.cleanse(x)
"""
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
    bsp = set(tuple(zip(mf.user_id, mf.friend_id))) #16,108
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
#df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')
from sklearn.externals.joblib import load, dump
from sklearn.neural_network import MLPClassifier
from importlib import reload
import func.cmodel as cmod
reload(cmod)

models = {
    'mlp' : MLPClassifier(hidden_layer_sizes=(50,), alpha=0.001, max_iter=100), #https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    
    'load' : load('./data/model/mlp.pkl')
    }
cmodel = cmod.cmodels(models['mlp'])
model = cmodel.train()
cmodel.eval()
dump(model, './data/model/mlp.pkl')
# df.to_hdf("./data/raw/dproc.h5", key='xx')

#%%--------------------------------------------
""" Recsys for all users 
[auc, hitrate, mrr] = recsysone(model, *(df, links))
"""
import func.cmodel as cmod
from importlib import reload
reload(cmod)

mlp = load('./data/model/mlp.pkl')
recsys = cmod.recsysone(mlp)
df = recsys.dfmanip(filtersize = 20, hsize=10)
[auc, hitrate, mrr] = recsys.eval()

#rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04')
#df_links = pd.read_hdf("./data/raw/dproc.h5", key='05')

#%%--------------------------------------------
""" Benchmark the results 
Save the results with config"""

#%%-------------------------------------------
""" Utility ops"""
def clearvars():
    import sys
    sys.modules[__name__].__dict__.clear()

