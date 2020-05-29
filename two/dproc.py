"""
Date: 02 May 2020
Goal: 01 Data processing for two.py
Author: Harsha HN harshahn@kth.se
"""

#%%----------------------
""" Load the network data """

class dproc:
    # Get node embeddings
    @staticmethod
    def getrawemb():
        #import pandas as pd
        # Load the feature data of each node, do feature engg data pre-processing such as scaling.
        # rawdf = pd.read_hdf("../data/common/dproc.h5", key='04').head()
        pass

    # Create DGL graph G from the network data
    @staticmethod
    def getdata(save = False):
        import pandas as pd
        ids = pd.read_hdf("../data/common/dproc.h5", key='04').index
        [trainids, trainpos, trainneg] = dproc.loadlinks() #, trainvnf
        print('-> Train set ids and links are finished.')

        # Validation set: friend links who are in trainids
        valpos = dproc.deltamf(trainpos, trainids)
        print('-> Validation set friend links are finished.')

        # Transform to new indices
        id_indx = dict(zip(trainids, range(len(trainids)))) #dict(enumerate(trainids))
        indx_id = dict(zip(range(len(trainids)), trainids))
        trainpos = [(id_indx[a], id_indx[b]) for a,b in trainpos]
        trainneg = [(id_indx[a], id_indx[b]) for a,b in trainneg]
        valpos = [(id_indx[a], id_indx[b]) for a,b in valpos]

        # Create DGL graph from friendship links.
        import dgl
        G = dproc.dglnx(len(trainids), trainpos)
        print('-> Graph G has %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 
        trainneg = list(zip(*trainneg)); trainpos = list(zip(*trainpos))
        #trainneg = set(tuple(zip(trainneg[0], trainneg[1]))) #trainpos = set(tuple(zip(trainpos[0], trainpos[1])))

        if save:
            import pickle
            with open('../data/two/check01.pkl', 'wb') as f:
                pickle.dump([G, trainpos, trainneg, valpos, id_indx, indx_id], f)
                print('-> Variables have been saved.')

        return [G, trainpos, trainneg, valpos, id_indx, indx_id]

    #%%----------------------------------
    # Create DGL graph from links and ids
    @staticmethod
    def dglnx(size, frds):
        import dgl
        import torch
        G = dgl.DGLGraph()
        G.add_nodes(size)
        G.add_edges(*zip(*frds))
        G = dgl.to_bidirected(G)
        #G.ndata['id'] = torch.tensor(ids)
        return G

    #%%-----------------------------
    # Calculate delta new links from train
    @staticmethod
    def deltamf(trainpos, trainids):
        import pandas as pd
        valmf = pd.read_hdf("../data/common/val/sqldata.h5", key='mf')
        #valaf = pd.read_hdf("../data/common/val/sqldata.h5", key='af')
        
        valmfs = set(tuple(zip(valmf.user_id, valmf.friend_id)))
        #valaf = valaf.groupby(['activity_id'])['user_id'].apply(list)
        #valafs = set()
        #for a,b in valaf.iteritems():
        #    valafs.update(set(itertools.combinations(b, 2)))

        valpos = valmfs #valpos = (valmfs | valafs) 
        links = set([(a,b) for (a,b) in valpos if (a in trainids) and (b in trainids)])
        dvalpos = links - set(trainpos)

        return dvalpos

    #%%
    # Load all the training set links from stored files
    @staticmethod
    def loadlinks():
        import pickle
        with open('../data/common/train.pkl', 'rb') as f:
            [posids, pos, neg] = pickle.load(f)
        return [posids, pos, neg]


    #%%
    @staticmethod
    def loadfresh(ids): #list(rawdf.index)
        import pandas as pd
        
        mf = pd.read_hdf("../data/common/network.h5", key='mf')
        af = pd.read_hdf("../data/common/network.h5", key='af')
        bf = pd.read_hdf("../data/common/network.h5", key='bf')
        vnf = pd.read_hdf("../data/common/network.h5", key='vnf')
        print('-> df files are loaded')

        mfs = set(tuple(zip(mf.user_id, mf.friend_id))) #16,108
        af = af.groupby(['activity_id'])['user_id'].apply(list)
        afs = set()
        for a,b in af.iteritems():
            afs.update(set(itertools.combinations(b, 2)))
        bfs = set(tuple(zip(bf.user_id, bf.blocked_id))) #13,684
        vnfs = set(tuple(zip(vnf.user_id, vnf.seen_id)))
        print('-> sets are created')

        def sub(subids, links):
            return set([(a,b) for (a,b) in links if (a in subids) and (b in subids)])

        submfs = sub(ids, mfs); subafs = sub(ids, afs); print('-> Pos finished')
        subbfs = sub(ids, bfs); subvnfs = sub(ids, vnfs); print('-> Neg finished')

        """
        import pickle
        with open('../data/common/sublinks.pkl', 'wb') as f:
            pickle.dump([submfs, subafs, subbfs, subvnfs], f)
        """
        """
        import pickle
        with open('../data/common/sublinks.pkl', 'rb') as f:
            [submfs, subafs, subbfs, subvnfs] = pickle.load(f)
        """
        
        allneg = (subbfs | subvnfs); pos = (submfs | subafs) - allneg

        import networkx as nx
        G = nx.Graph()
        G.add_edges_from(pos)
        posids = list(G.nodes())
        neg = set([(a,b) for (a,b) in allneg if (a in posids) and (b in posids)])
        """
        import pickle
        with open('../data/common/train.pkl', 'wb') as f:
            pickle.dump([posids, pos, neg], f)
        """    
        return [posids, pos, neg]
