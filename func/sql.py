
import os
import pandas as pd
import pymysql

#%% 
def df_sqlquery(sqlquery):
    print('-> Openning SQL connection...')
    db = pymysql.connect( "127.0.0.1", "root", "gofrendly", "gofrendly" ) # Open database connection
    print('-> Connection established at: 127.0.0.1, root, gofrendly, gofrendly')
    print('--> SQL query begins...')
    df = pd.read_sql_query(sqlquery, db)
    print('--> Extraction finished ')
    db.close()
    print('-> SQL connection is closed ')
    return df
