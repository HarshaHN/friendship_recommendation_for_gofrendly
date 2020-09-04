#%%----------------------------
""" Import libs """
import pandas as pd
import pymysql
from sql import sqlquery

#%%----------------------------
""" SQL data extraction """
def newsqldataext():
    queries = {
                # Mutually connected friends(hard positive)
                # count: a.120,409(119,750) b.128,138 (122,768) c.148,407 (141,865) # dbeaver(wb)
                'mf' : "SELECT a.user_id, a.friend_id FROM friends a\
                INNER JOIN friends b ON (a.user_id = b.friend_id) AND (a.friend_id = b.user_id) WHERE (a.user_id > a.friend_id)",
                #'af' : "SELECT activity_id, user_id FROM activity_invites WHERE isGoing = 1",
            }
    print('--> SQL query begins...')
    
    # gofrendly, gofrendly-api, test (148,407)

    with sqlquery('gofrendly-api') as newconn:
        mf = newconn.query(queries['mf']) 
        #mf.to_hdf("../data/common/test/all_network.h5", key='mf')
        print('--> mf\' query finished ')
        #af = newconn.query(queries['af']) 
        #af.to_hdf("../data/common/val/af.h5", key='af')
        #print('--> af\' query finished ')
    return mf

mf = newsqldataext()


# %%
