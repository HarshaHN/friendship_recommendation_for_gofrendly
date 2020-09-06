#%%-------------------
""" 01. Google Translator API client """
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"gcp_key.json"

from google.cloud import translate_v2 as translate
client = translate.Client()

#%%-------------------
""" 02. Story translations with GCP Translator client """
import pandas as pd

df = pd.read_hdf("../data/one/cleansed_stories.h5", key='01')
batch_size = 10; loops = len(df)//batch_size +1

#%%-----------
import time

for i in range(1400, loops): # loops
        t0 = time.time()
        snip_df = df[i*batch_size:(i+1)*batch_size]
        print('Translation loop:',i,'begins...')
        snip_df['story_english'] = snip_df['story'].apply(lambda x: client.translate(x, target_language='en')['translatedText'] if x!=-1 else -1)
        print('Translation loop:',i,'complete.')
        snip_df.to_hdf("stories/snip.h5", key=str(i))
        print('Translation loop:',i,'saved. Timed:', time.time()-t0) # ~1s

# 0, 1000 one
# 1000,1400 two
# 1400, loops(1495) three
#%%-----------------------
""" 03. Merge all the user story into one dataframe """
path = "stories/snip.h5" #one, two, three
#full_df = pd.read_hdf(path, key=str(0))
for i in range(1, loops): # loops
        snip_df = pd.read_hdf(path, key=str(i))
        full_df = pd.concat([full_df, snip_df], axis=0)

#%%------------------------
full_df.to_hdf("../data/stories/stories_sv_en.h5", key='01')
fulldf = full_df.drop(columns=['story']); fulldf.columns = ['story']
fulldf.to_hdf("../data/stories/stories.h5", key='01')
