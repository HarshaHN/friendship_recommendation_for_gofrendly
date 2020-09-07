"""
File: 02_query_(test_data).py
To get friendship connections from sql data for the purpose of model performance evaluation. This is exclusively for validation and test data.
"""

#%%
import pandas as pd
import pymysql
from sql_conn import sqlquery

#%%
def friendship_connections():
    queries = {
                # Mutually connected friends (hard positive)
                'mf' : "SELECT a.user_id, a.friend_id FROM friends a\
                INNER JOIN friends b ON (a.user_id = b.friend_id) AND (a.friend_id = b.user_id)\
                WHERE (a.user_id > a.friend_id)"
            }
    print('--> SQL query begins...')
    
    # dbname = gofrendly (train), gofrendly-api (val)
    dbname = 'gofrendly-api'
    with sqlquery(dbname) as newconn:
        mf = newconn.query(queries['mf']) 
        # mf.to_hdf("out/xxx/all_network.h5", key='mf') # xxx: val or test.
        print('--> mf query finished ')
    return mf

mf = friendship_connections()
