
import os
import pandas as pd
import pymysql

#%% 
def sqlconnect(): #Open SQL connection
    print('Openning SQL connection...', '\n')
    db = pymysql.connect( "127.0.0.1", "root", "gofrendly", "gofrendly" ) # Open database connection
    print('SQL connection returned at: 127.0.0.1, root, gofrendly, gofrendly \n')
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
        "SELECT activity_id, user_id FROM activity_invites where isGoing = 1",
        # 5. Chat links

        """ Classification model """
        """a. Positive samples"""
        # 1. Chat friends(hard positive)
        # 2. Mutually connected friends(hard positive)
        # 3. Activity friends(soft positive)
        # 4. Chat friends of chat friends(very soft positive)
        # Comments: 1 and 2 may overlap, 3 shows common interest & may lead to more 
        # of those, 4 can be populated however may never have seen each other.

        """b. Negative samples"""
        # 1. Blocked user pairs (hard negative) 
        # count: a. b.13,684 c. 
        'SELECT user_id, blocked_id FROM blocked_users',
        # 2. Viewed users but not added as friends (hard negative)
        # Viewed users count: a. b.4,846,799 c. 
        'SELECT user_id, seen_id FROM seen_users'
        # 3. one-way added as friend but not chat friends (hard negative)
        # one-way friends count: a. b.1,102,338 c.
        #Comments: 

        #Overall comments:
        # A = A-B leading to mutually exclusive groups.
        ]

    os.chdir('./data/raw')
    #global uNodes, fLinks, aNodes, aLinks
    uNodes = pd.read_sql_query(query[0], db)
    fLinks = pd.read_sql_query(query[1], db)
    aLinks = pd.read_sql_query(query[3], db)
    aNodes = pd.read_sql_query(query[2], db)
    #cLinks = pd.read_sql_query(query[4], db)
    saveone([uNodes, fLinks, aNodes, aLinks])
    db.close() # disconnect from server 
    print('SQL query is complete and data has been saved!')

def saveone(listpd): #save the dfs
    os.chdir('./data/raw')
    [uNodes, fLinks, aNodes, aLinks] = listpd
    uNodes.to_hdf("uNodes.h5", key='uNodes')
    fLinks.to_hdf("fLinks.h5", key='fLinks')
    aNodes.to_hdf("aNodes.h5", key='aNodes')
    aLinks.to_hdf("aLinks.h5", key='aLinks')
    #cLinks.to_hdf("cLinks.h5", key='cLinks')
    os.chdir('../..')
