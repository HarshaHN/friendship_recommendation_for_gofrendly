#%%-------------
"""
Date: 02 March 2020
Goal: 01 Data processing for one.py
Author: Harsha HN harshahn@kth.se
"""

""" 01. Data Pre-processing """
import pickle
import itertools
import numpy as np
import pandas as pd
import dgl

class dproc:

    # Load all the links of training dataset
    @staticmethod
    def trainlinks():
        mf = pd.read_hdf("../data/common/train/network.h5", key='mf')
        af = pd.read_hdf("../data/common/train/network.h5", key='af')
        bf = pd.read_hdf("../data/common/train/network.h5", key='bf')
        vnf = pd.read_hdf("../data/common/train/network.h5", key='vnf')
        print('-> Training df files are loaded')
        return [mf, af, bf, vnf]

    # Load all the links of validation dataset
    @staticmethod
    def vallinks():
        mf = pd.read_hdf("../data/common/val/network.h5", key='mf')
        #af = pd.read_hdf("../data/common/val/network.h5", key='af')
        #bf = pd.read_hdf("../data/common/val/network.h5", key='bf')
        #vnf = pd.read_hdf("../data/common/val/network.h5", key='vnf')
        print('-> Validation df files are loaded')
        return [mf]#, af]#, bf, vnf] 

    """
    # Load all the links of test dataset
    @staticmethod
    def testlinks(): 
        mf = pd.read_hdf("../data/common/test/network.h5", key='mf')
        af = pd.read_hdf("../data/common/test/network.h5", key='af')
        #bf = pd.read_hdf("../data/common/test/network.h5", key='bf')
        #vnf = pd.read_hdf("../data/common/test/network.h5", key='vnf')
        print('-> Test df files are loaded')
        return [mf, af]#, bf, vnf]    
    """    
    # Load all the links from stored files
    @staticmethod
    def makepairs(links, trainflag):
        if trainflag==1:
            [mf, af, bf, vnf]  = links
            mfs = set(tuple(zip(mf.user_id, mf.friend_id))) #16,108
            af = af.groupby(['activity_id'])['user_id'].apply(list)
            afs = set()
            for a,b in af.iteritems():
                afs.update(set(itertools.combinations(b, 2)))
            bfs = set(tuple(zip(bf.user_id, bf.blocked_id))) #13,684
            vnfs = set(tuple(zip(vnf.user_id, vnf.seen_id)))
            print('-> Pairs have been made')
            return [mfs, afs, bfs, vnfs] 

        elif trainflag==0:
            [mf, af]  = links
            mfs = set(tuple(zip(mf.user_id, mf.friend_id))) #16,108
            af = af.groupby(['activity_id'])['user_id'].apply(list)
            afs = set()
            for a,b in af.iteritems():
                afs.update(set(itertools.combinations(b, 2)))
            print('-> Pairs have been made')
            return [mfs, afs] 

    # Filter relevant links
    @staticmethod
    def sublinks(ids, links, trainflag):
        def sub(links, subids):
            return set([(a,b) for (a,b) in links if ((a in subids) and (b in subids))])
        
        if trainflag==1:
            [mfs, afs, bfs, vnfs] = links        
            submfs, subafs, subbfs, subvnfs = sub(mfs, ids), sub(afs, ids), sub(bfs, ids), sub(vnfs, ids)
            neg = (subbfs | subvnfs); pos = (submfs | subafs) - neg
            return [list(pos), list(neg)]
        elif trainflag==0:
            [mfs, afs] = links        
            submfs, subafs = sub(mfs, ids), sub(afs, ids)
            pos = (submfs | subafs)
            return list(pos)

    # Training links
    @staticmethod
    def getlinks(ids):
        # Training links.
        [trainpos, trainneg] = dproc.sublinks(ids, dproc.makepairs(dproc.trainlinks(), trainflag=1), trainflag=1)
        print('-> 01 Trainings links are captured.')

        # Validation links
        #valpos = dproc.sublinks(ids, dproc.makepairs(dproc.vallinks(), trainflag=0), trainflag=0)
        #valpos = set(valpos) - set(trainpos)
        print('-> 02 Validation links are captured.')

        # Test links
        #testpos = dproc.sublinks(ids, dproc.makepairs(dproc.testlinks(), trainflag=0), trainflag=0)
        #testpos = set(testpos) - set(valpos) - set(trainpos)
        print('-> 03 Test links are captured.')

        return [trainpos, trainneg]#, valpos]#, testpos

    # Data pre-processing
    @staticmethod
    def preproc(df):
        # df = ['user_id', 'story', 'iam', 'meetfor', 'birthday', 'marital', 'children', 'lat', 'lng']
        df.columns = ['id', 'story', 'iam', 'meetfor', 'age', 'marital', 'kids', 'lat', 'lng']
        df.set_index('id', inplace = True)
        
        """ 01. Data cleanse and imputation """

        #'iam', 'meetfor' to set()
        df['iam'] = df['iam'].apply(lambda x: list(x.split(',')) if ((x!=None) and (len(x)>0)) else -1).astype('int32')
        df['meetfor'] = df['meetfor'].apply(lambda x: list(x.split(',')) if ((x!=None) and (len(x)>0)) else -1).astype('int32')
        
        #'birthday' to age
        df['age'] = df['age'].apply(lambda x: int((x.today() - x).days/365)).clip(18, 100).astype('int32')
        
        # has children, marital
        df['marital'] = df['marital'].fillna('-1').astype('int32') # df.marital.value_counts().keys()
        df['kids'] = df['kids'].fillna('-1').astype('int32')       # df.kids.value_counts().keys()

        # story
        mycleanse = cleanse()
        df['story'] = df['story'].apply(lambda x: mycleanse.cleanse(x))
        
        # df.to_hdf("../data/one/trainfeat.h5", key='01')

        """ 02. stories translation with GCP """
        # df['story'] = pd.read_hdf("../data/one/stories.h5", key='01')
        
        """ 03. Stories to S-BERT emb """
        df['emb'] = df['story'].apply(lambda x: np.random.randn(1204) if x!=-1 else -1)
        # df['emb'] = pd.read_hdf("../data/one/emb.h5", key='01')
        #df.drop(columns=['story'])

        #df.to_hdf("../data/one/trainfeat.h5", key='02')
        return df
    
    # Feature Engineering: categorical and numerical features
    @staticmethod
    def feature(feat):        
        # import pandas as pd; import numpy as np
        # feat = pd.read_hdf("../data/one/trainfeat.h5", key='02')
        
        # 01. Numerical data        
        from sklearn.preprocessing import robust_scale

        feat.age = robust_scale(feat.age.to_numpy()[:, None])
        feat.lat = robust_scale(feat.lat.to_numpy()[:, None])
        feat.lng = robust_scale(feat.lng.to_numpy()[:, None])

        feat['num'] = feat.index
        feat['num'] = feat['num'].apply(lambda x: [feat.age[x], feat.lat[x], feat.lng[x]])
        feat = feat.drop(columns = ['age', 'lat', 'lng'])

        # 02. Categorical data
        def onehotencode(input, dim):
            onehot = np.zeros(dim, dtype=int)
            try:
                if isinstance(input, (int, np.integer)):
                    onehot[input] = 1
                else: 
                    for el in input:
                        ind = int(el)
                        if ind < dim: onehot[ind] = 1
            except:
                print(input)
            return onehot

        feat['cat'] = feat.index
        feat['cat'] = feat['cat'].apply(lambda x: np.concatenate(( onehotencode(feat.iam[x], 18), onehotencode(feat.meetfor[x], 18), onehotencode(feat.marital[x], 5), onehotencode(feat.kids[x], 4) )))
        feat = feat.drop(columns = ['iam', 'meetfor', 'marital', 'kids', 'story'])

        #feat.to_hdf("../data/one/trainfeat.h5", key='03') # ['emb', 'cat', 'num']
        return df

    # Load the training links
    @staticmethod
    def loadlinks():
        import pickle
        # import pandas as pd
        # feat = pd.read_hdf("../data/one/trainfeat.h5", key='03') # ['emb', 'cat', 'num'] #dproc: preproc >> feature
        # [trainpos, trainneg] = dproc.getlinks(feat.index) 
        # with open('../data/one/sublinks.pkl', 'wb') as f: # links.pickle
        #    pickle.dump([trainpos, trainneg], f) #402761, 72382 

        with open('../data/one/sublinks.pkl', 'rb') as f: # links.pickle
            [trainpos, trainneg] = pickle.load(f) #402761, 72382
        return [trainpos, trainneg]

    # DGL graph
    @staticmethod
    def makedgl(num, pos):
        G = dgl.DGLGraph()
        G.add_nodes(num)
        G.add_edges(G.nodes(), G.nodes()) #self loop all
        G.add_edges(*zip(*pos)) #add edges list(zip(*pos))
        G = dgl.to_bidirected(G) 
        print('-> Graph G has %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 
        return G        


#%%-------------
""" 02. Functional """
# Cleanse the text
import re
import emoji #conda install -c conda-forge emoji
class cleanse:
    #cleanse story
    r = re.compile(r'([.,/#!?$%^&*;:{}=_`~()-])[.,/#!?$%^&*;:{}=_`~()-]+')

    @classmethod
    def cleanse(cls, text):
        if (text == '') or (text == None): #.isnull()
            text = -1
        else:
            text = text.replace("\n", ". ") #remove breaks
            text = emoji.get_emoji_regexp().sub(u'', text) #remove emojis
            text = cls.r.sub(r'\1', text) #multiple punctuations
            if len(text) < 10: 
                text = -1 #short texts
        return text
