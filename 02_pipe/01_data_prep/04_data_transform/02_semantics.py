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
""" 02. Generate SBERT embeddings"""
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
sbert_df = pd.read_hdf("out/stories/semantics_data.h5", key='01')
