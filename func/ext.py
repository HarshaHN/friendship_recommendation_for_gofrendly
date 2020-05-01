#%%
import pandas as pd
import pymysql
import func.sql as opsql

#%%
""" Exploratory Data Analysis """
# Perform EDA from pandas
# https://medium.com/datadriveninvestor/introduction-to-exploratory-data-analysis-682eb64063ff
# Use data pre-processing libs such as 
query = { 
        #User profile features
        'in' : "SELECT user_id, iAm, meetFor, birthday, marital, children FROM user_details \
        INNER JOIN users ON user_details.user_id=users.id",
        'info' : "SELECT user_id, birthday, city, country, lat, lng FROM user_details \
        INNER JOIN users ON user_details.user_id=users.id"
    }
df = opsql.df_sqlquery(query['01'])
df.to_hdf("./data/raw/ext.h5", key='01')

del query 
#info.to_hdf("./data/raw/in.h5", key='info')

#%%
""" 2. Classification model """
def cmodel():
    query = {
            # a. Positive samples
            # 1. Chat friends(hard positive)

            # 2. Mutually connected friends(hard positive)
            # count: a. b.120,409 c. 
            'mf' : "SELECT a.user_id, a.friend_id FROM friends a\
            INNER JOIN friends b ON (a.user_id = b.friend_id) AND (a.friend_id = b.user_id) WHERE (a.user_id > a.friend_id)",

            # 3. Activity friends(soft positive)
            # count: a. b.32,422 c. 
            'af' : "SELECT activity_id, user_id FROM activity_invites WHERE isGoing = 1",

            # 4. Chat friends of chat friends(soft positive)
            # Comments: 1 and 2 may overlap, 3 shows common interest & may lead to more 
            # of those, 4 can be populated however may never have seen each other.

            #b. Negative samples
            # 1. Blocked user pairs (hard negative) 
            # count: a. b.13,684 c. 
            'bf' : "SELECT user_id, blocked_id FROM blocked_users",

            # 2. Viewed users but not added as friends (hard negative)
            # Viewed users count: a. b.4,464,793(4,846,799) c. 
            'vnf' : "SELECT a.user_id, a.seen_id FROM seen_users a\
            LEFT JOIN friends b\
            ON (b.user_id = a.user_id) AND (b.friend_id = a.seen_id)\
            WHERE b.user_id IS null",

            # 3. one-way added as friend but not chat friends (hard negative)
            # one-way friends count: a. b.1,102,338 c.
            'uf' : "SELECT user_id, friend_id FROM friends"#,
            #Comments: 
            # A = A-B leading to mutually exclusive groups.
        }
    # mf, af, bf, vnf, uf
    print('--> SQL query begins...')
    
    mf = opsql.df_sqlquery(query['mf'])
    mf.to_hdf("./data/raw/cmodel.h5", key='mf')
    print('--> mf\' query finished ')
    af = opsql.df_sqlquery(query['af'])
    af.to_hdf("./data/raw/cmodel.h5", key='af')
    print('--> af\' query finished ')
    bf = opsql.df_sqlquery(query['bf'])
    bf.to_hdf("./data/raw/cmodel.h5", key='bf')
    print('--> bf\' query finished ')
    vnf = opsql.df_sqlquery(query['vnf'])
    vnf.to_hdf("./data/raw/cmodel.h5", key='vnf')
    print('--> vnf\' query finished ')
    uf = opsql.df_sqlquery(query['uf'])
    uf.to_hdf("./data/raw/cmodel.h5", key='uf')
    print('--> uf\' query finished ')

    db.close()
    print('db connection has been closed.')
    return [mf, af, bf, vnf, uf]

[mf, af, bf, vnf, uf] = cmodel()

#%%
import pandas as pd
import itertools

""" Data prep for C model """
users = pd.read_hdf("./data/raw/in.h5", key='info')

mf = pd.read_hdf("./data/raw/cmodel.h5", key='mf')
af = pd.read_hdf("./data/raw/cmodel.h5", key='af')

bf = pd.read_hdf("./data/raw/cmodel.h5", key='bf')
vnf = pd.read_hdf("./data/raw/cmodel.h5", key='vnf')
uf = pd.read_hdf("./data/raw/cmodel.h5", key='uf')


#%%
#pdata = pd.DataFrame()
#ndata = pd.DataFrame()

# CF: 1
#asetplus = 

# MF: 1, count: a. b.120409 c. 
bsp = set(tuple(zip(mf.user_id, mf.friend_id)))
#bplus = pd.DataFrame({ 'pairs': tuple(zip(mf.user_id, mf.friend_id)), 'out': [1]*mf.shape[0]})
#pdata = pdata.append(bplus, ignore_index = True)

# AF: 1, count: a. b.32422 c. 
af = af.groupby(['activity_id'])['user_id'].apply(list)
csp = set()
for a,b in af.iteritems():
    csp.update(set(itertools.combinations(b, 2)))
#cplus = pd.DataFrame({ 'pairs': list(csetplus), 'out': [1]*len(csetplus)})
#pdata = pdata.append(cplus, ignore_index = True)

# CF of CF: 1, count: a. b. c.
#dsetplus = 

# BF:0, count: a. b.13,684 c. 
asm = set(tuple(zip(bf.user_id, bf.blocked_id)))
#temp = pd.DataFrame({ 'pairs': tuple(zip(bf.user_id, bf.blocked_id)), 'out': [0]*bf.shape[0]})
#ndata = ndata.append(temp, ignore_index = True)

# VNF:0, count: a. b.4,464,793 c. 
bsm = set(tuple(zip(vnf.user_id, vnf.seen_id)))
#temp = pd.DataFrame({ 'pairs': tuple(zip(vnf.user_id, vnf.seen_id)), 'out': [0]*vnf.shape[0]})
#ndata = ndata.append(temp, ignore_index = True)

# UF:0
#csetminus = 

del a, b, mf, af, bf, vnf, uf


#%%
""" 3. Network aggregation """

#%% I/O data architect

#%% Data flow architect

#%% Compatibility to data evolution

#%% backup or reference
