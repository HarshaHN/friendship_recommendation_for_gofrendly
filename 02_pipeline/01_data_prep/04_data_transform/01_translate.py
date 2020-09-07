#%%
"""
File: translate.py
To translate the user stories to English.
"""

#%%
""" 01. Google Translator API client """
import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"gcp_key.json"

from google.cloud import translate_v2 as translate
client = translate.Client()

#%%
""" 02. Setup input """
import pandas as pd

df = pd.DataFrame(pd.read_hdf('../03_data_cleansing/out/userinfo_cleaned.h5', key='dcleanse').story) #stockholm_users_df
num_chars = df.story.apply(lambda x: len(x) if x!=-1 else 0).sum(0) # 3,947,808 characters
cost = num_chars*20/1000000-10; print('cost =', round(cost),'USD or', round(cost*9.33),'SEK')
# cost = 69.0 USD or 643.0 SEK

batch_size = 10; loops = len(df)//batch_size +1 #1495

#%%
""" 03. Translation """
import time

for i in range(loops):
    t0 = time.time()
    snip_df = df[i*batch_size:(i+1)*batch_size]
    print('Translation loop:', i,'begins...')
    snip_df['story_english'] = snip_df['story'].apply(lambda x: client.translate(x, target_language='en')['translatedText'] if x!=-1 else -1)
    print('Translation loop:', i,'complete.')
    snip_df.to_hdf("out/stories/snip.h5", key=str(i))
    print('Translation loop:', i,'saved. Timed:', time.time()-t0) # ~1s

#%%
""" 04. Merge the user stories into one dataframe """
path = "out/stories/snip.h5"
full_df = pd.read_hdf(path, key=str(0))

for i in range(1, loops): # loops
        snip_df = pd.read_hdf(path, key=str(i))
        full_df = pd.concat([full_df, snip_df], axis=0)

#%%
""" 05. Save the stories """
full_df.to_hdf("out/stories/stories_sv_eng.h5", key='01')
fulldf = full_df.drop(columns=['story']); fulldf.columns = ['story']
fulldf.to_hdf(".out/stories/stories_eng.h5", key='01')
