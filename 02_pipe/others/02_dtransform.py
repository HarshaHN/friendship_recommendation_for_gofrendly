#%%
"""
File: dtransform.py
To transform the data into numerical tensors.
"""

#%%
import pickle
import pandas as pd
import re
import emoji #conda install -c conda-forge emoji

class dtransform:



# DGL graph
@staticmethod
def makedgl(num, pos):
    G = dgl.DGLGraph()
    G.add_nodes(num)
    G.add_edges(G.nodes(), G.nodes()) #self loop all
    G.add_edges(*zip(*pos)) #add edges list(zip(*pos))
    G = dgl.to_bidirected(G) 
    G = dgl.graph(G.edges(), 'user', 'frd')
    print('-> Graph G has %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 
    return G  








#%%
"""
File: sbert.py
To generate SBERT embeddings for user stories.
"""

#%%
""" 01. Import data """
import pandas as pd
stories_df = pd.read_hdf("out/stories/stories_eng.h5", key='01')

#%%---------------------
""" 02. SBERT """
from tqdm import tqdm
tqdm.pandas()

from sentence_transformers import SentenceTransformer
sbertmodel = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

def encode(x): 
    return sbertmodel.encode([x])[0]

before = time.time()
stories_df['emb'] = stories_df['story'].progress_apply(lambda x: encode(x) if x!=-1 else -1)
print("-> S-BERT embedding finished.", (time.time() - before)) #6000s

#%%
"""03. Save the files """
stories_df.drop(columns = 'story', inplace = True)
sbert_df = pd.read_hdf("out/stories/sbert_emb.h5", key='01')






    @staticmethod
    def process(df):
        # ['index', 'id', 'story', 'iam', 'meetfor', 'age', 'marital', 'kids', 'lat', 'lng']
        
        """ cleanse and imputation """
        #'iam', 'meetfor'
        df['iam'] = df['iam'].apply(lambda x: list(x.split(',')) if ((x!=None) and (len(x)>0)) else -1)
        df['meetfor'] = df['meetfor'].apply(lambda x: list(x.split(',')) if ((x!=None) and (len(x)>0)) else -1)
        
        #'birthday' to age
        df['age'] = df['age'].apply(lambda x: int((x.today() - x).days/365)).clip(18, 100).astype('int32')
        
        # has kids, marital
        df['marital'] = df['marital'].fillna('-1').astype('int32') # df.marital.value_counts().keys()
        df['kids'] = df['kids'].fillna('-1').astype('int32')       # df.kids.value_counts().keys()

        # story
        df['story'] = df['story'].apply(lambda x: dcleanse.textcleanse(x))
        print('-> User info has been cleaned!')

        return df

#%% To execute class dcleanse
cleanse_op = dcleanse()
stockholm_users_df = pd.read_hdf("../02_data_organization/out/stockholm_users.h5", key='dorg')
df = cleanse_op.process(stockholm_users_df)
df.to_hdf('out/userinfo_cleaned.h5', key='dcleanse')

#%% To execute class dorg
cleanse_op = dcleanse()
stockholm_users_df = pd.read_hdf("../02_data_organization/out/stockholm_users.h5", key='dorg')
cleanse_op.process(stockholm_users_df)

#org_op.train_links(list(stockholm_users_df.user_id)) #saved as out/train_links.pkl
#org_op.train_userinfo(stockholm_users_df)            #saved as out/stockholm_users.h5", key='dorg'

#%%
# df.to_hdf("../data/one/stories_emb.h5", key='01')
#feat_df = pd.read_hdf("../data/one/trainfeat.h5", key='04') 
#sbert_df = pd.read_hdf("../data/one/sbert_emb.h5", key='01')
#feat_df['emb'] = sbert_df['emb']
#feat_df.to_hdf("../data/one/user_features.h5", key='02') # ['emb', 'cat', 'num']


#%%
uf = pd.read_hdf("../one/user_features.h5", key='01')
f = pd.read_hdf("../one/user_features.h5", key='02')
uf.to_hdf('out/userinfo_cleaned.h5', key='dcleanse')
#path = '../one/user_features.h5'; import h5py; f = h5py.File(path, 'r'); [key for key in f.keys()]



#%%

#%%
import itertools
import numpy as np
import dgl


""" 02. stories translation with GCP """
# df['story'] = pd.read_hdf("../data/one/stories.h5", key='01')

""" 03. Stories to S-BERT emb """
df['emb'] = df['story'].apply(lambda x: np.random.randn(1024) if x!=-1 else -1)
# df['emb'] = pd.read_hdf("../data/one/emb.h5", key='01')
#df.drop(columns=['story'])

#df.to_hdf("../data/one/trainfeat.h5", key='02')
return df

# Feature Engineering: categorical and numerical features
@staticmethod
def feature(feat):        
    # import pandas as pd; import numpy as np
    # feat = pd.read_hdf("../data/one/trainfeat.h5", key='02')
    
    # 01. Numerical data        

    feat['num'] = feat.index
    feat['num'] = feat['num'].apply(lambda x: [feat.age[x], feat.lat[x], feat.lng[x]])
    feat = feat.drop(columns = ['age', 'lat', 'lng'])

    # 02. Categorical data
    def onehotencode(input, dim):
        onehot = np.zeros(dim, dtype=int)
        try:
            if isinstance(input, (int, np.integer)):
                onehot[input] = 1
            else: 
                for el in input:
                    ind = int(el)
                    if ind < dim: onehot[ind] = 1
        except:
            print(input)
        return onehot

    feat['cat'] = feat.index
    feat['cat'] = feat['cat'].apply(lambda x: np.concatenate(( onehotencode(feat.iam[x], 18), onehotencode(feat.meetfor[x], 18), onehotencode(feat.marital[x], 5), onehotencode(feat.kids[x], 4) )))
    feat = feat.drop(columns = ['iam', 'meetfor', 'marital', 'kids', 'story'])

    #feat.to_hdf("../data/one/trainfeat.h5", key='04') # ['emb', 'cat', 'num']
    return df

# Load the training links
@staticmethod
def loadlinks():
    import pickle
    # import pandas as pd
    # feat = pd.read_hdf("../data/one/trainfeat.h5", key='03') # ['emb', 'cat', 'num'] #dproc: preproc >> feature
    # [trainpos, trainneg] = dproc.getlinks(feat.index) 

    with open('../data/one/rawidx_nw.pkl', 'rb') as f: # links.pickle
        [trainpos, trainneg] = pickle.load(f) #402761, 72382
    return [trainpos, trainneg]



# %%
