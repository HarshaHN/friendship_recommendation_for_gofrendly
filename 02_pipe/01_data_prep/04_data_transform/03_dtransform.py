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
