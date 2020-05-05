
import os
import pandas as pd
import pymysql
#%%
class sqlquery():

    def __enter__(self):
        print('-> Open SQL connection...')
        self.db = pymysql.connect( "127.0.0.1", "root", "gofrendly", "gofrendly" ) # Open database connection
        print('-> Connection established at: 127.0.0.1, root, gofrendly, gofrendly')
        return self

    def query(self, q):
        print('--> SQL query begins...')
        df = pd.read_sql_query(q, self.db)
        print('--> Extraction finished ')
        return df

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # make sure the dbconnection gets closed
        self.db.close()

#with sqlquery() as newconn:
