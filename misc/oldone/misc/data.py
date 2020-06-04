#%%-----------------------
""" import libraries """
import pandas as pd
import time

#%%-------------
""" Data Pre-processing """
class dproc:
    """ ['user_id', 'myStory', 'iAm', 'meetFor', 'birthday', 'marital', 'children', 'lat', 'lng'] ->  """
    df = pd.read_hdf("./data/raw/dproc.h5", key='01')

    def dfmanip(self):
        #[ user_id, myStory, values(iAm, meetFor, marital, has child, age, lat, lng) ]
        df.set_index('user_id', inplace = True)
        
        """ 01. Data cleanse and imputation """
        # myStory
        mycleanse = cleanse()
        df['myStory'] = df['myStory'].apply(lambda x: mycleanse.cleanse(x))
        #'iAm', 'meetFor' to set()
        df['iAm'] = df['iAm'].apply(lambda x: set(x.split(',')) if x != None else set())
        df['meetFor'] = df['meetFor'].apply(lambda x: set(x.split(',')) if x != None else set())
        #'birthday' to age
        df['age'] = df['birthday'].apply(lambda x: int((x.today() - x).days/365))
        df.drop(columns='birthday', inplace = True)
        # has children, marital
        df['marital'].fillna(-1, inplace=True)
        df['children'].fillna(-1, inplace=True)
        #df.to_hdf("./data/raw/dproc.h5", key='02')
        #user vectors = [ user_id, 'myStory', 'iAm', 'meetFor', 'marital', 'children', 'lat', 'lng', 'age']

        """ 02. stories translation with GCP """
        df['myStory'] = df['myStory'].apply(lambda x: gcptrans(x) if x!=-1 else -1)
        #df.to_hdf("./data/raw/dproc.h5", key='03')

        """ 03. Stories to S-BERT emb """
        from sentence_transformers import SentenceTransformer
        sbertmodel = SentenceTransformer('roberta-large-nli-mean-tokens')

        def listup(x):
            listx = list()
            listx.append(x)
            return listx
        before = time.time()
        df['emb'] = df['myStory'].apply(lambda x: sbertmodel.encode(listup(x)) if x!=-1 else -1)
        #df['emb'] = [user_emb['emb'][0], user_emb['emb'][1]]*7474
        #df['emb'] = df['emb'].apply(lambda x: x.reshape(1,-1))
        print("-> S-BERT embedding finished.", (time.time() - before)) #534 sec
        df.drop(columns = 'myStory', inplace = True)
        #df.to_hdf("./data/raw/dproc.h5", key='04')
        #user vectors = [ user_id, 'iAm', 'meetFor', 'marital', 'children', 'lat', 'lng', 'age', 'emb']

        """04. Consolidate all the links """ #Only once
        #ids = list(df.index); [p, n] = self.links(ids)
        def getlinks():    
            import pickle
            filename = './data/vars/links.pickle'
            with open(filename, 'rb') as f:
                return pickle.load(f) #[m, p]
            return -1
        [neg, pos] = getlinks()

        """05. Compute delta of two users """
        # Links to delta 'in'
        df_pos = pd.DataFrame({'links': list(pos)})#.head()
        df_neg = pd.DataFrame({'links': list(neg)})#.head()

        df_pos['in'] = df_pos['links'].apply(lambda x: delta(*x))
        df_pos['out'] = [1]*len(df_pos)
        print('--> Finished for positive links')
        df_neg['in'] = df_neg['links'].apply(lambda x: delta(*x))
        df_neg['out'] = [0]*len(df_neg)
        print('--> Finished for negative links')

        ## SMOTE ##
        #posX = df_pos['in']; posY = df_pos['out']
        #negX = df_neg['in']; negY = df_neg['out']
        # X = posX + negX # Y = posY + negY  

        df_links = pd.concat([df_pos, df_neg], ignore_index=True) #del df_pos, df_neg
        #X = df_links['in]; Y = df_links['out']
        #df_links.to_hdf("./data/raw/dproc.h5", key='05')
        
        return df
    
     #   def links(ids):

            import itertools
            """ Links extraction from the network """
            #positive samples
            mf = pd.read_hdf("./data/raw/cmodel.h5", key='mf')
            af = pd.read_hdf("./data/raw/cmodel.h5", key='af')
            #negative samples
            bf = pd.read_hdf("./data/raw/cmodel.h5", key='bf')
            vnf = pd.read_hdf("./data/raw/cmodel.h5", key='vnf')
            print("01 -- vars loaded!")

            #mf as bsp
            bsp = set(tuple(zip(mf.user_id, mf.friend_id))) #16,108
            #af as csp
            af = af.groupby(['activity_id'])['user_id'].apply(list)
            csp = set()
            for a in af.iteritems():
                csp.update(set(itertools.combinations(a[1], 2)))

            #bf as asm
            asm = set(tuple(zip(bf.user_id, bf.blocked_id)))
            #vnf as bsm
            bsm = set(tuple(zip(vnf.user_id, vnf.seen_id)))
            print("02 -- links compiled!")

            del mf, af, bf, vnf

            """ Links subset """
            def sublinks(subids, allids):
                temp = set()
                for a,b in allids:
                    if (a in subids) and (b in subids):
                        temp.add((a,b))
                return temp

            am = sublinks(ids, asm); bm = sublinks(ids, bsm); m = (am | bm)
            ap = sublinks(ids, bsp); cp = sublinks(ids, csp); p = (ap | cp) - m
            print("03 -- classes created!")

            del bsp, csp, asm, bsm, am, ap, cp, bm

            #Save the vars
            import pickle
            filename = './data/vars/links.pickle'
            with open(filename, 'wb') as f:
                pickle.dump([m, p], f)  
            del f, filename

            return [p, m]


#%%-------------
""" Cleanse myStory """
import re
import emoji #conda install -c conda-forge emoji
class cleanse:
    #cleanse myStory
    r = re.compile(r'([.,/#!?$%^&*;:{}=_`~()-])[.,/#!?$%^&*;:{}=_`~()-]+')

    @classmethod
    def cleanse(cls, text):
        if (text == '') or (text == None): #.isnull()
            text = -1
        else:
            text = text.replace("\n", ". ") #remove breaks
            text = emoji.get_emoji_regexp().sub(u'',text) #remove emojis
            text = cls.r.sub(r'\1', text) #multiple punctuations
            if len(text) < 10: 
                text = -1 #short texts
        return text

#%%--------------------
""" GCP translator API """
def gcptrans(text):
    #Corpus with example sentences
    texts = [ 'A man is eating food.',
                'A man is eating a piece of bread.',
                'The girl is carrying a baby.',
                'A man is riding a horse.',
                'A woman is playing violin.',
                'Two men pushed carts through the woods.',
                'A man is riding a white horse on an enclosed ground.',
                'A monkey is playing drums.',
                'A cheetah is running behind its prey.']
    import random
    return texts[random.randint(0,8)]


#%%----------------

def misc():
    pass
    """
    import h5py
    f = h5py.File('../data/common/eda/location.h5', 'r')
    [key for key in f.keys()]
    """
    #%%
    """
    # CF: 1
    #asetplus = 

    # MF: 1, count: a. b.120409(16,108) c. 
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


    #%% I/O data architect

    #%% Data flow architect

    #%% Compatibility to data evolution

    #%% backup or reference
    """