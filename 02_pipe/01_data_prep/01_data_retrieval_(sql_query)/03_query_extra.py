"""
File: 03_query_extra.py
For experimental purpose only. 
"""

#%%
import pandas as pd
import pymysql
from sql_conn import sqlquery

#%%
# User profile data
def userdata():
    queries = { 
        #User profile
        'location' : "SELECT user_id, birthday, city, country, lat, lng FROM user_details \
        INNER JOIN users ON user_details.user_id=users.id",
        'all_users_profile' : "SELECT user_id, myStory, iAm, meetFor, birthday, marital, children, lat, lng FROM user_details a INNER JOIN users b ON (a.user_id=b.id)"
    }

    # dbname = gofrendly (train), gofrendly-api (val)
    dbname = 'gofrendly'
    with sqlquery(dbname) as newconn:
        info = newconn.query(queries['location']) 
        # info.to_hdf("out/extra/all_location.h5", key='info')
        print('--> location query finished ')

        df = newconn.query(queries['all_users_profile']) 
        # df.to_hdf("./05_data_store/01_out_(data_retrieval)/extra/all_sqlusers.h5", key='01')
        print('--> all_users_profile query finished ')

        del queries
    return df

#%%
# User network connection
def sqllinks():
    queries = {
            # A. Positive samples
            # 1. Chat friends (hard positive)

            # 2. Mutually connected friends (hard positive)
            'mf' : "SELECT a.user_id, a.friend_id FROM friends a\
            INNER JOIN friends b ON (a.user_id = b.friend_id) AND (a.friend_id = b.user_id) WHERE (a.user_id > a.friend_id)",

            # 3. Activity friends(soft positive)
            # count: a.32,422 (train) b. (val) c. (test)
            'af' : "SELECT activity_id, user_id FROM activity_invites WHERE isGoing = 1",

            # 4. Chat friends of chat friends(soft positive)
            # Comments: 
            # 1 and 2 may overlap, 3 shows common interest & may lead to more 
            # of those, 4 can be populated however may never have seen each other.

            # B. Negative samples
            # 1. Blocked user pairs (hard negative) 
            # count: a.13,684 b. c. 
            'bf' : "SELECT user_id, blocked_id FROM blocked_users",

            # 2. Viewed users but not added as friends (soft negative)
            # Viewed users count: a.4,464,793(4,846,799) b. c. 
            'vnf' : "SELECT a.user_id, a.seen_id FROM seen_users a\
            LEFT JOIN friends b\
            ON (b.user_id = a.user_id) AND (b.friend_id = a.seen_id)\
            WHERE b.user_id IS null",

            # 3. one-way added as friend but not chat friends (hard negative)
            # one-way friends count: a. b.1,102,338 c.
            #'uf' : "SELECT user_id, friend_id FROM friends"#,
            
            #Comments: 
            # A = A-B leading to mutually exclusive groups.
        }
    # mf, af, bf, vnf
    print('--> SQL query begins...')
    
    # gofrendly, gofrendly-api
    with sqlquery('gofrendly') as newconn:        

        # Positive samples
        mf = newconn.query(queries['mf']) 
        #mf.to_hdf("../data/common/extra/network.h5", key='mf')
        print('--> mf\' query finished ')
        af = newconn.query(queries['af']) 
        #af.to_hdf("../data/common/extra/network.h5", key='af')
        print('--> af\' query finished ')

        # Negative samples
        bf = newconn.query(queries['bf'])
        #bf.to_hdf("../data/common/extra/network.h5", key='bf')
        print('--> bf\' query finished ')
        vnf = newconn.query(queries['vnf']) 
        #vnf.to_hdf("../data/common/extra/network.h5", key='vnf')
        print('--> vnf\' query finished ')

    return [mf, af, bf, vnf]

#[mf, af, bf, vnf] = sqllinks()

#%%
""" To load the data """
def load():
    location = pd.read_hdf("out/extra/all_location.h5", key='info')
    users = pd.read_hdf("out/extra/all_sqlusers.h5", key='01')

    mf = pd.read_hdf("out/extra/network.h5", key='mf')
    af = pd.read_hdf("out/extra/network.h5", key='af')
    bf = pd.read_hdf("out/extra/network.h5", key='bf')
    vnf = pd.read_hdf("out/extra/network.h5", key='vnf')

    return [users, mf, af, bf, vnf] 
