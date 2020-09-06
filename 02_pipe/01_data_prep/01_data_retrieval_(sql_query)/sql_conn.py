"""
File: sql_conn.py
To manage database connection so to carry-out sql query.
"""

#%%
import os
import pandas as pd
import pymysql

#%%
class sqlquery():

    def __init__(self, data):
        self.database = data

    def __enter__(self):
        # open db connection to sql data
        print('-> Open SQL connection...')
        self.db = pymysql.connect( "127.0.0.1", "root", "gofrendly", self.database ) # Open database connection
        print('-> Connection established at: 127.0.0.1, root, gofrendly', self.database)
        return self
        
    def query(self, q):
        # perform sql query
        print('--> SQL query begins...')
        df = pd.read_sql_query(q, self.db)
        print('--> Extraction finished ')
        return df

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # make sure the dbconnection gets closed
        self.db.close()
