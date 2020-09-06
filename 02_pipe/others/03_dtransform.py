#%%
"""
File: dtransform.py
To transform the data into numerical tensors.
"""

#%%
import pickle
import pandas as pd
import numpy as np
import torch 

class dtransform:

    @staticmethod
    def process(df):
        # 01. Numerical data
        df['num'] = df.index
        df['num'] = df['num'].apply(lambda x: [df.age[x], df.lat[x], df.lng[x]])
        df = df.drop(columns = ['age', 'lat', 'lng'])
        print('--> 01 Numerical data has been processed!')

        # 02. Categorical data
        def onehotencode(input, dim):
            onehot = np.zeros(dim, dtype=int)
            try:
                if isinstance(input, (int, np.integer)):
                    onehot[input] = 1
                else: 
                    for ind in input:
                        ind = int(ind)
                        if ind < dim: onehot[ind] = 1
            except:
                print('Error for', input)
            return onehot
        df['cat'] = df.index
        df['cat'] = df['cat'].apply(lambda x: np.concatenate(( onehotencode(df.iam[x], 18), onehotencode(df.meetfor[x], 18), onehotencode(df.marital[x], 5), onehotencode(df.kids[x], 4) )))
        df = df.drop(columns = ['iam', 'meetfor', 'marital', 'kids', 'story'])
        print('--> 02 Categorical data has been processed!')

        # 03. Semantics data
        sem_df = pd.read_hdf("out/semantics_data.h5", key='01')
        sem_df['emb'] = sem_df['emb'].apply(lambda x: np.zeros(768) if type(x)==int else x)
        df['sem'] = sem_df.reset_index(drop=True)
        print('--> 03 Semantic data has been processed!')
        
        return df
    
    @staticmethod
    def tensors(feat_df):
        # Transform numerical features into tensors
        num_data = torch.tensor(list(feat_df['num']), dtype=torch.float32)
        cat_data = torch.tensor(list(feat_df['cat']), dtype=torch.float32)
        sem_data = torch.tensor(list(feat_df['sem']), dtype=torch.float32)
        usr_feat = torch.cat((num_data, cat_data, sem_data), 1)
        return usr_feat

#%% To execute class dtransform
transform_obj = dtransform()
df = pd.read_hdf('../03_data_cleansing/out/userinfo_cleaned.h5', key='dcleanse') #stockholm_users_df
df = transform_obj.process(df)
usr_feat = transform_obj.tensors(df)

# save the tensor
torch.save(usr_feat, 'out/user_tensor.pt')


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


#%% To execute class dcleanse
cleanse_op = dcleanse()
stockholm_users_df = pd.read_hdf("../02_data_organization/out/stockholm_users.h5", key='dorg')
df = cleanse_op.process(stockholm_users_df)

#df.to_hdf('out/userinfo_cleaned.h5', key='dcleanse')
#org_op.train_links(list(stockholm_users_df.user_id)) #saved as out/train_links.pkl
#org_op.train_userinfo(stockholm_users_df)            #saved as out/stockholm_users.h5", key='dorg'
#df.to_hdf("../data/one/stories_emb.h5", key='01')
#feat_df = pd.read_hdf("../data/one/trainfeat.h5", key='04') 
#sbert_df = pd.read_hdf("../data/one/sbert_emb.h5", key='01')
#feat_df['emb'] = sbert_df['emb']
#feat_df.to_hdf("../data/one/user_features.h5", key='02') # ['emb', 'cat', 'num']
uf = pd.read_hdf("../one/user_features.h5", key='01')
f = pd.read_hdf("../one/user_features.h5", key='02')
uf.to_hdf('out/userinfo_cleaned.h5', key='dcleanse')
#path = '../one/user_features.h5'; import h5py; f = h5py.File(path, 'r'); [key for key in f.keys()]





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

