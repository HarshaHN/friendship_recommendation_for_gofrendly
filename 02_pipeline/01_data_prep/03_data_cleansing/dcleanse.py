#%%
"""
File: dcleanse.py
To clean the data and includes imputation, text cleansing. 
"""

#%%
import pickle
import pandas as pd
import re
import emoji #conda install -c conda-forge emoji


class dcleanse:
    # multiple punctuation symbols to remove
    r = re.compile(r'([.,/#!?$%^&*;:{}=_`~()-])[.,/#!?$%^&*;:{}=_`~()-]+')

    @classmethod
    def textcleanse(cls, text):
        if (text == '') or (text == None): #.isnull()
            text = -1
        else:
            text = text.replace("\n", ". ") #to remove line breaks
            text = emoji.get_emoji_regexp().sub(u'', text) #to remove emojis
            text = cls.r.sub(r'\1', text) #to remove multiple punctuations
            if len(text) < 10: 
                text = -1 #to remove short texts
        return text

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
