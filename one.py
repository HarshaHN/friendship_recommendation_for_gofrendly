"""
Date: 24 Feb 2020
Author: Harsha harshahn@kth.se
Friendship link prediction using deep neural network (Social Network Analysis) as a part of Master thesis
"""

# %%
"""import libraries"""
# Tools
import os
import numpy as np
import pandas as pd
import networkx as nx

# SQL
import pymysql

# ML libraries


#%% 
"""SQL query"""
db = pymysql.connect( "127.0.0.1", "gofrendly", "gofrendly", "gofrendly" ) # Open database connection

#db.close() # disconnect from server


# %% 
"""Data Pre-processing"""
#Extract raw data through Sequel query
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

uNodes = pd.read_sql_query(query[0], db)
fLinks = pd.read_sql_query(query[1], db)
aLinks = pd.read_sql_query(query[3], db)
aNodes = pd.read_sql_query(query[2], db)
#cLinks = pd.read_sql_query(query[4], db)

#lemmatize
#Swedish and English
#uNodes: mystory, describeyourself, iamcustom, meetforcustom #iAm, meetFor
#aNodes: title, description FROM activities

#remove stop words


#POS tagging and keep nouns and verbs or not?

#Extract chat connections from Firebase
# (userid, chatfriend) = cLinks

uNodes.to_hdf("uNodes.h5", key='uNodes')
fLinks.to_hdf("fLinks.h5", key='fLinks')
aNodes.to_hdf("aNodes.h5", key='aNodes')
aLinks.to_hdf("aLinks.h5", key='aLinks')
#cLinks.to_hdf("cLinks.h5", key='cLinks')

os.chdir('../..')
db.close() # disconnect from server 


# %%
"""Data Analysis"""
os.chdir('./data/in')
uNodes = pd.read_hdf("uNodes.h5", key='uNodes')
fLinks = pd.read_hdf("fLinks.h5", key='fLinks')
aNodes = pd.read_hdf("aNodes.h5", key='aNodes')
aLinks = pd.read_hdf("aLinks.h5", key='aLinks')
#cLinks = pd.read_hdf("cLinks", key='cLinks')
os.chdir('../..')

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
