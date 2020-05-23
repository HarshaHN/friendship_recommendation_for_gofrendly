
#%%----------------------
""" Load the network data """
import torch.nn as nn

def getrawemb(nodes):
    #Load the feature data of each node, do data pre-processing such as scaling.
    #Add features to the DGL graph
    #import pandas as pd
    #rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04').head()

    embed = nn.Embedding(nodes, 2)  # 34 nodes with embedding dim equal to 5
    return embed.weight

def deltamf(trainmf, trainids, id_idx): #trainmf, trainids, id_idx
    # trainmf, trainids, id_idx >> deltavalmf(id_idx) which are in trainids
    import pandas as pd
    valmf = pd.read_hdf("./data/02_val/sqldata.h5", key='mf')
    valmfs = set(tuple(zip(valmf.user_id, valmf.friend_id)))

    links = set()
    for a,b in valmfs:
        if (a in trainids) and (b in trainids):
            links.add((a,b))
    delta = list(links - set(trainmf))
    
    dvalmf = list()
    for a,b in delta:
        s = id_idx[a]; d = id_idx[b]
        dvalmf.append((s,d))

    return dvalmf

#%%
def loadmf():
    #loads the mf ids
    import pandas as pd
    rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04')
    mf = pd.read_hdf("./data/raw/cmodel.h5", key='mf')
    bsp = set(tuple(zip(mf.user_id, mf.friend_id))) #16,108

    def sublinks(subids, allids):
        temp = set(); graphids = list()
        for a,b in allids:
            if (a in subids) and (b in subids):
                temp.add((a,b))
                if a not in graphids: graphids.append(a)
                if b not in graphids: graphids.append(b)
        return [graphids, list(temp)]
    ids = list(rawdf.index)
    return [ids, sublinks(ids, bsp)]

def dglnx(ids, id_idx, mf):
    import dgl
    import torch
    G = dgl.DGLGraph(); frds = list()
    G.add_nodes(len(ids))
    G.ndata['id'] = torch.tensor(ids)
    for (a,b) in mf:
        s = id_idx[a]; d = id_idx[b]
        frds.append((s,d))
        G.add_edge(s, d); G.add_edge(d, s)
    return [frds, G]

def nxviz(G):
    import networkx as nx
    nx_G = G.to_networkx()#.to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
