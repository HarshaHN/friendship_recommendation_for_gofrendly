#%%
"""
File: dorg.py
To organize the data  to includes rearrange, merge, format, separate and unify into relevant data frame.
"""

#%%
import pickle
import pandas as pd

class dorg:

    @staticmethod
    def train_links(city_userids):
        # Load network data
        def load_nwdata():
            mf = pd.read_hdf("../01_data_retrieval_(sql_query)/out/train/all_network.h5", key='mf')
            bf = pd.read_hdf("../01_data_retrieval_(sql_query)/out/train/all_network.h5", key='bf')
            print('-->01 Network data are loaded')
            return [mf, bf]

        # Network data into network pairs
        def make_nwpairs(nw_data):
            [mf, bf]  = nw_data
            mfs = set(tuple(zip(mf.user_id, mf.friend_id))) #120,409
            bfs = set(tuple(zip(bf.user_id, bf.blocked_id))) #13,684
            print('-->02 Pairs have been made')
            return [mfs, bfs]

        # Limit network pairs to one city
        def filt_pairs(city_userids, nw_pairs):
            
            # Retain pairs of city users alone
            def sub(nw_pairs, city_userids):
                return set([(a,b) for (a,b) in nw_pairs if ((a in city_userids) and (b in city_userids))])
            
            [mfs, bfs] = nw_pairs
            neg = sub(bfs, city_userids); pos = sub(mfs, city_userids) - neg
            print('-->03 Pairs have been filtered')
            return [list(pos), list(neg)]

        # Re-index the positive and negative links
        def getlinks(city_userids):
            [p, n] = filt_pairs(city_userids, make_nwpairs(load_nwdata()))
            id_idx = {id: n for n,id in enumerate(city_userids)}
            # re-index
            pos = [(id_idx[a], id_idx[b]) for a,b in p]
            neg = [(id_idx[a], id_idx[b]) for a,b in n]

            with open('out/train_links.pkl', 'wb') as f: pickle.dump([pos, neg], f)
            print('-->04 Training links have been re-indexed and saved.')
        
        getlinks(city_userids)
        print('->01 Train links have been organized.')

    @staticmethod
    def train_userinfo(stockholm_users_df):
        stockholm_users_df.columns = ['id', 'story', 'iam', 'meetfor', 'age', 'marital', 'kids', 'lat', 'lng']
        stockholm_users_df.to_hdf("out/stockholm_users.h5", key='dorg')
        print('->02 Training user information has been organized and saved.')

#%% To execute class dorg
org_op = dorg()
stockholm_users_df = pd.read_hdf("../01_data_retrieval_(sql_query)/out/train/stockholm_users.h5", key='01')
# ['index', 'user_id', 'myStory', 'iAm', 'meetFor', 'birthday', 'marital', 'children', 'lat', 'lng']
org_op.train_links(list(stockholm_users_df.user_id)) #saved as out/train_links.pkl
org_op.train_userinfo(stockholm_users_df)            #saved as out/stockholm_users.h5", key='dorg'
