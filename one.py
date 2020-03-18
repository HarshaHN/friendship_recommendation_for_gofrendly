"""
Date: 24 Feb 2020
Author: Harsha harshahn@kth.se
Friendship recommendation engine using deep neural network (Social Network Analysis) as a part of Master thesis
"""

# %%
"""import libraries"""
#def importlibs():
    # Tools
import os
import numpy as np
import pandas as pd
import networkx as nx
import time

    # ML libraries

# %% Clear all variables
def clearvars():
    import sys
    sys.modules[__name__].__dict__.clear()

#%% 
def sqlconnect(): #Open SQL connection
    print('Openning SQL connection...', '\n')
    import pymysql
    db = pymysql.connect( "127.0.0.1", "gofrendly", "gofrendly", "gofrendly" ) # Open database connection
    print('SQL connection returned at: 127.0.0.1, gofrendly, gofrendly, gofrendly \n')
    #db.close() # disconnect from server
    return db

def save_sqlquery(db): #Run SQL query and extract data
    print('Begin SQL query...', '\n')
    query = [
        # 1. Select user profile data 
        "SELECT user_id, isActive, isProfileCompleted, lang, myStory, describeYourself, iAmCustom, meetForCustom, iAm, meetFor, marital, children \
            FROM user_details \
                INNER JOIN users ON user_details.user_id=users.id",
        # 2. Select friend links
        "SELECT user_id, friend_id FROM friends", 
        # 3. Activities data
        "SELECT id, title, description FROM activities",
        # 4. Activity links
        "SELECT activity_id, user_id FROM activity_invites"#,
        # 5. Chat links
        ]

    os.chdir('./data/in')
    global uNodes, fLinks, aNodes, aLinks
    uNodes = pd.read_sql_query(query[0], db)
    fLinks = pd.read_sql_query(query[1], db)
    aLinks = pd.read_sql_query(query[3], db)
    aNodes = pd.read_sql_query(query[2], db)
    #cLinks = pd.read_sql_query(query[4], db)
    saveone()
    db.close() # disconnect from server 
    print('SQL query is complete and data has been saved!')

#%%
from func.sql import sqlconnect, save_sqlquery
#global uNodes, fLinks, aNodes, aLinks
db = sqlconnect()
save_sqlquery(db)

# %% 
# Save the files
def saveone(): #save the dfs
    os.chdir('./data/in')
    global uNodes, fLinks, aNodes, aLinks
    uNodes.to_hdf("uNodes.h5", key='uNodes')
    fLinks.to_hdf("fLinks.h5", key='fLinks')
    aNodes.to_hdf("aNodes.h5", key='aNodes')
    aLinks.to_hdf("aLinks.h5", key='aLinks')
    #cLinks.to_hdf("cLinks.h5", key='cLinks')
    os.chdir('../..')

def loadone(): #load the dfs
    os.chdir('./data/in')
    global uNodes, fLinks, aNodes, aLinks
    uNodes = pd.read_hdf("uNodes.h5", key='uNodes')
    #fLinks = pd.read_hdf("fLinks.h5", key='fLinks')
    #aNodes = pd.read_hdf("aNodes.h5", key='aNodes')
    #aLinks = pd.read_hdf("aLinks.h5", key='aLinks')
    #cLinks = pd.read_hdf("cLinks", key='cLinks')
    os.chdir('../..')

# %%
#Translate the text to english
def trans(sub_stories):
    from googletrans import Translator
    translator = Translator()
    langlist =[]; #langdist = []
    for index, row in sub_stories.iterrows():
        text = row['myStory'].replace("\n", ' ')
        if (text != None) and (text != ''):
            #langdist.append(translator.detect(row).lang)
            try: 
                langlist.append(translator.translate(text, dest = 'en').text)
            except: print( index, '\n', text , '\n', '###'); break;
        else: langlist.append(None)
    return langlist

#from matplotlib import pyplot as plt
#plt.hist(langlist)

# %%
"""Data processing"""
loadone() #load the dfs

#a. Lang translation to english {German, Swedish, Norwegian} 
#global uNodes, fLinks, aNodes, aLinks
stories = pd.concat([uNodes['user_id'], uNodes['myStory']], axis=1)

subStories = stories[:10] # take out a sample
subStories['myStory_en'] = trans(subStories) # translate
subStories.to_hdf("subStories.h5", key='stories') #save them
subStories = pd.read_hdf("subStories.h5", key='stories') #load them

#%%
subStories = pd.read_hdf("stories.h5", key='stories') #load 
def removenull(text):
    text = text[~text['myStory'].isnull()]
    text = text[text['myStory'] != '']
    return text
subStories = removenull(subStories)
del subStories['myStory']

# %%
#Sentence Embeddings using BERT / RoBERTa / XLNet https://pypi.org/project/sentence-transformers/
from sentence_transformers import SentenceTransformer
# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('bert-base-nli-mean-tokens')

stories = list(subStories['myStory_en'])
userId = list(subStories['user_id'])

stories_embeddings = model.encode(stories)
    
queries = stories[5:6]
query_embeddings = model.encode(queries)

start_time = time.time()

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(stories_embeddings[:5])

print(neigh.kneighbors(query_embeddings))
print("--- %s seconds ---" % (time.time() - start_time))

#%%

#a. textual data: myStory. BERT embeddings. Tokenize sentences, average BERT embeddings.

#lemmatize
#Swedish and English

#remove stop words

#POS tagging and keep nouns and verbs or not?

#Extract chat connections from Firebase
#start_time = time.time()
#print("--- %s seconds ---" % (time.time() - start_time))

# (userid, chatfriend) = cLinks

# %%
"""Data Analysis"""
loadone() #load the dfs
#Build the Heterogenous Social Network Graph
#G = nx.Graph()


# %%
#Evaluation
#print("Hello there!")


# %%
#User interaction and display


# %%
#Training

# %%
#save and commit

# %%
#main


# %%
# Machine Learning based

# %%
# Deep Learning based
