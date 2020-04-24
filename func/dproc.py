
#%%
import pandas as pd
import pymysql
import func.sql as opsql

# For 'city = Göteborg', get [ user_id, stories, iam, meetFor, birthday, marital, children, lat, lng]
# df = df_sqlquery(query)
query = {
    '01' : "SELECT user_id, myStory, iAm, meetFor, birthday, marital, children, lat, lng FROM user_details a\
            INNER JOIN users b ON (a.user_id=b.id) WHERE a.city = \"Stockholm\"",
    }
df = opsql.df_sqlquery(query['01'])
del query 
df.to_hdf("./data/raw/dproc.h5", key='01')
#------------------------------------------------------------------------------------

#%%
df = pd.read_hdf("./data/raw/dproc.h5", key='01')
# [ user_id, myStory, values(iAm, meetFor, marital, has child, age, lat, lng) in range(0,1) ]
# from ['user_id', 'myStory', 'iAm', 'meetFor', 'age', 'marital', 'children', 'lat', 'lng']

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
    #except: print(text)
    return text

df['myStory'] = df['myStory'].apply(lambda x: cleanse(x))

#'iAm', 'meetFor' to set()
df['iAm'] = df['iAm'].apply(lambda x: set(x.split(',')) if x != None else set())
df['meetFor'] = df['meetFor'].apply(lambda x: set(x.split(',')) if x != None else set())

#'birthday' to age
df['age'] = df['birthday'].apply(lambda x: int((x.today() - x).days/365))
df.drop(columns='birthday', inplace = True)

# has children, marital
#df['has_kids'] = df['children'].apply(lambda x: 1 if (x>=0) else -1)

df.to_hdf("./data/raw/dproc.h5", key='02')
#--------------------------------------------

#%% stories translation with GCP
df = pd.read_hdf("./data/raw/dproc.h5", key='02')

# df.to_hdf("./data/raw/dproc.h5", key='03')
#--------------------------------------------

# Stories to S-BERT emb

# df.to_hdf("./data/raw/dproc.h5", key='04')
#--------------------------------------------

# user vectors = [ user_id, sbert_emb, values(iam, meetFor, marital, has children), eucld_diff(age, lat, lng) in range(0,1) ]

# df.to_hdf("./data/raw/dproc.h5", key='05')
#--------------------------------------------

# delta(uv1, uv2) = [ cosine_sim_sbert, count(intersection(iam, meetFor)/3) binary_equality(marital, has children), eucld_diff(age, lat, lng) in range(0,1) ]

# df.to_hdf("./data/raw/dproc.h5", key='06')
#--------------------------------------------

# train c-model using delta vectors.

# df.to_hdf("./data/raw/dproc.h5", key='07')
#--------------------------------------------

# Benchmark the results

# df.to_hdf("./data/raw/dproc.h5", key='08')
#--------------------------------------------













"""
'in' : "SELECT user_id, iAm, meetFor, birthday, marital, children FROM user_details \
INNER JOIN users ON user_details.user_id=users.id",
main =  pd.read_hdf("./data/raw/in.h5", key='in')

'info' : "SELECT user_id, birthday, city, country, lat, lng FROM user_details 
main =  pd.read_hdf("./data/raw/in.h5", key='info')

"""

# %%
