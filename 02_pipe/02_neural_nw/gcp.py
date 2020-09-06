
#%%-------------------
""" 01. stories translation with GCP """
import pandas as pd
#df = pd.read_hdf("../data/one/trainfeat.h5", key='01')
#df = df.drop(columns=['iam', 'meetfor', 'age', 'marital', 'kids', 'lat', 'lng'])
#df.to_hdf("../data/one/cleansed_stories.h5", key='01')

df = pd.read_hdf("../data/one/cleansed_stories.h5", key='01')
num_chars = df.story.apply(lambda x: len(x) if x!=-1 else 0).sum(0) #3,947,808
cost = num_chars*20/1000000-10; print('cost =', round(cost),'USD or', round(cost*9.33),'SEK')
# cost = 69.0 USD or 643.0 SEK

#text = df.story.loc[657]

#%%-------------------
""" 02. Google Translator API """
# Refer to translate.py
stories_df = pd.read_hdf("../data/one/stories.h5", key='01')

#%%---------------------
""" 03. SBERT """
from tqdm import tqdm
tqdm.pandas()

from sentence_transformers import SentenceTransformer
sbertmodel = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

df = pd.read_hdf("../data/one/stories.h5", key='01')
def encode(x): return sbertmodel.encode([x])[0]

before = time.time()
df['emb'] = df['story'].progress_apply(lambda x: encode(x) if x!=-1 else -1)
print("-> S-BERT embedding finished.", (time.time() - before)) #6000s

# df.drop(columns = 'story', inplace = True)
# df.to_hdf("../data/one/stories_emb.h5", key='01')
df = pd.read_hdf("../data/one/stories_emb.h5", key='01')


#feat_df = pd.read_hdf("../data/one/trainfeat.h5", key='04') 
#sbert_df = pd.read_hdf("../data/one/sbert_emb.h5", key='01')
#feat_df['emb'] = sbert_df['emb']
#feat_df.to_hdf("../data/one/user_features.h5", key='02') # ['emb', 'cat', 'num']