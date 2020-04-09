
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
        "SELECT activity_id, user_id FROM activity_invites where isGoing = 1"#,
        # 5. Chat links
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
