
#%%----------------------
""" Load the network data """

def loadmf():
    import pandas as pd
    rawdf = pd.read_hdf("./data/raw/dproc.h5", key='04')
    mf = pd.read_hdf("./data/raw/cmodel.h5", key='mf')
    bsp = set(tuple(zip(mf.user_id, mf.friend_id))) #16,108

    def sublinks(subids, allids):
        temp = set()
        for a,b in allids:
            if (a in subids) and (b in subids):
                temp.add((a,b))
        return temp
    ids = list(rawdf.index)
    return [ids, list(sublinks(ids, bsp))]

def dglnx(ids, id_idx, mf):
    import dgl
    import torch
    G = dgl.DGLGraph(); frds = list()
    G.add_nodes(len(ids))
    G.ndata['id'] = torch.tensor(ids)
    for (a,b) in mf:
        s = id_idx[a]; d = id_idx[b]
        frds.append([s,d])
        G.add_edge(s, d); G.add_edge(d, s)
    return [frds, G]
