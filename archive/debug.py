# To add a new cell, type ''
# To add a new markdown cell, type ' [markdown]'

"""
Date: 24 Feb 2020
Author: Harsha harshahn@kth.se
Friendship link prediction using deep neural network methods(Social Network Analysis) as a part of Master thesis
"""


"""import libraries"""
# Tools
import numpy as np
import pandas as pd
import networkx as nx
#import matplotlib.pyplot as plt

# SQL
import pymysql
#import pymysql.err as errorcode

# ML libraries
#from sklearn import model_selection
#import sklearn.metrics
#

 
"""For debug run"""
"""
import mysql.connector
from mysql.connector import errorcode

try:
    cnx = mysql.connector.connect(user='gofrendly', password='gofrendly', host='127.0.0.1',  database='gofrendly')
except mysql.connector.Error as err:
  if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
    print("Something is wrong with your user name or password")
  elif err.errno == errorcode.ER_BAD_DB_ERROR:
    print("Database does not exist")
  else:
    print(err)
else:
  cnx.close()
"""

 
"""SQL query"""
db = pymysql.connect("127.0.0.1","gofrendly","gofrendly","gofrendly" ) # Open database connection
cursor = db.cursor() # prepare a cursor object using cursor() method
query =  "SELECT table_name FROM information_schema.tables"
cursor.execute(query) # execute SQL query using execute() method.
data = cursor.fetchall()

#cursor.close() #close cursor
#db.close() # disconnect from server

 
"""Data Processing"""
query = open('read.sql', 'r')
#cursor.execute(query)
tables = pd.read_sql(query.read(), db)

#Build the Heterogenous Social Network Graph
#G = nx.Graph()
                        

#Data Analysis


#Evaluation
#print("Hello there!")



#User interaction and display



#Training


#save and commit


#main


# Statistics-based


# Machine Learning based


# Deep Learning based
