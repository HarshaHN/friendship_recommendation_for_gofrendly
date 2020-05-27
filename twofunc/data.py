
#%%----------------------
""" Load the network data """
import torch

def getemb(nodes, fdim=5, features = False):
    import torch.nn as nn
    #Load the feature data of each node, do feature engg data pre-processing such as scaling.
    #import pandas as pd
    #rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04').head()
    embed = nn.Embedding(nodes, fdim) 
    return embed.weight

def createG(save = False):

    [ids, trainids, trainmf, trainbf] = loadlinks() #, trainvnf
    print('-> Train set mf and bf links are finished.')

    id_indx = dict(zip(trainids, range(len(trainids)))) #dict(enumerate(trainids))
    indx_id = dict(zip(range(len(trainids)), trainids))

    # Create DGL graph from friendship links.
    import dgl
    [trainfrds, G] = dglnx(trainids, id_indx, trainmf)
    print('-> Graph G has %d nodes' % G.number_of_nodes(), 'with %d edges' % (G.number_of_edges()/2)) 

    trainblk = [(id_indx[a], id_indx[b]) for a,b in trainbf]
    trainneg = list(zip(*trainblk))
    trainpos = list(zip(*trainfrds))

    # Validation set: friend links who are in trainids
    valpos = deltamf(trainmf, trainids, id_indx)
    print('-> Validation set friend links are finished.')

    if save:
        # Save
        import pickle
        with open('./data/two/check01.pkl', 'wb') as f:
            pickle.dump([G, trainpos, trainneg, valpos, id_indx, indx_id], f)
            print('-> Variables have been saved.')

    #del trainblk, trainbf, trainfrds, trainmf, ids, trainids
    return [G, trainpos, trainneg, valpos, id_indx, indx_id]

def deltamf(trainmf, trainids, id_indx): #trainmf, trainids, id_indx
    import pandas as pd
    valmf = pd.read_hdf("./data/02_val/sqldata.h5", key='mf')
    valmfs = set(tuple(zip(valmf.user_id, valmf.friend_id)))

    links = set()
    for a,b in valmfs:
        if (a in trainids) and (b in trainids):
            links.add((a,b))
    delta = list(links - set(trainmf))

    dvalmf = [(id_indx[a], id_indx[b]) for a,b in delta]

    return dvalmf

#%%
def dglnx(ids, id_indx, mf):
    import dgl
    import torch
    G = dgl.DGLGraph(); frds = list()
    G.add_nodes(len(ids))
    frds = [(id_indx[a], id_indx[b]) for a,b in mf]
    G.add_edges(*zip(*frds))
    G = dgl.to_bidirected(G)
    G.ndata['id'] = torch.tensor(ids)
    return [frds, G]

#%%
def loadlinks():
    #loads the mf with its ids, bf and all ids.
    import pandas as pd
    rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04')
    mf = pd.read_hdf("./data/raw/cmodel.h5", key='mf')
    mfs = set(tuple(zip(mf.user_id, mf.friend_id))) #16,108
    bf = pd.read_hdf("./data/raw/cmodel.h5", key='bf')
    bfs = set(tuple(zip(bf.user_id, bf.blocked_id))) #13,684
    #vnf = pd.read_hdf("./data/raw/cmodel.h5", key='vnf')
    #vnfs = set(tuple(zip(vnf.user_id, vnf.seen_id)))
    mfs = mfs - bfs
    print('-> df files are loaded')

    def sub(subids, links, lids=0):
        sublinks = set(); linkids = list()
        for a,b in links:
            if (a in subids) and (b in subids):
                sublinks.add((a,b))
                if lids == 1:
                    if a not in linkids: linkids.append(a)
                    if b not in linkids: linkids.append(b)
        return [linkids, list(sublinks)]
    ids = list(rawdf.index)
    [mids, mf] = sub(ids, mfs, lids=1); print('-> mf finished')
    [_, bf] = sub(mids, bfs); print('-> bf finished')
    #[_, vnf] = sub(mids, vnfs); print('-> vnf finished')
    return [ids, mids, mf, bf]

def nxviz(G):
    import networkx as nx
    nx_G = G.to_networkx()#.to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
