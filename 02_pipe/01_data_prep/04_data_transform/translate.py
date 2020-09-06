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
""" 02. Story translations with GCP Translator client """
import pandas as pd

stockholm_users_df = pd.read_hdf('out/userinfo_cleaned.h5', key='dcleanse')

df = pd.read_hdf("../data/one/cleansed_stories.h5", key='01')
batch_size = 10; loops = len(df)//batch_size +1

#%%
""" 03. Translation """
import time

for i in range(loops): # loops
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


""" 02. stories translation with GCP """
# df['story'] = pd.read_hdf("../data/one/stories.h5", key='01')

#%%
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

""" 02. Google Translator API """
# Refer to translate.py
stories_df = pd.read_hdf("../data/one/stories.h5", key='01')
