"""
File: 01_query_(train_data).py
To get user profile information and friendship connections from sql data. This is exclusivly for training data.
"""

#%%
import pandas as pd
import pymysql
from sql_conn import sqlquery

#%%
def userdata():
    queries = { 
        # User profile for Stockholm
        'user_profile' : "SELECT user_id, myStory, iAm, meetFor, birthday, marital, children, lat, lng FROM user_details a INNER JOIN users b ON (a.user_id=b.id) WHERE a.city = \"Stockholm\"",

        # Positive samples
        # Mutually connected friends (hard positive)
        # count: a. (train), b. (val), c. (test)
        'mf' : "SELECT a.user_id, a.friend_id FROM friends a\
        INNER JOIN friends b ON (a.user_id = b.friend_id) AND (a.friend_id = b.user_id)\
        WHERE (a.user_id > a.friend_id)",

        # Negative samples
        # Blocked user pairs (hard negative) 
        # count: a.13,684 (train), b. (val), c. (test)
        'bf' : "SELECT user_id, blocked_id FROM blocked_users",
    }

    # dbname = gofrendly (train), gofrendly-api (val)
    dbname = 'gofrendly'
    with sqlquery(dbname) as newconn:
        df = newconn.query(queries['user_profile']) 
        # df.to_hdf("out/train/stockholm_users.h5", key='01')
        print('--> user_profile query finished ')

        mf = newconn.query(queries['mf']) 
        # mf.to_hdf("out/train/all_network.h5", key='mf')
        print('--> mf query finished ')

        bf = newconn.query(queries['bf']) 
        # bf.to_hdf("out/train/all_network.h5", key='bf')
        print('--> bf query finished ')
        
        del queries
    return [df, mf, bf]

[df, mf, bf] = userdata()

