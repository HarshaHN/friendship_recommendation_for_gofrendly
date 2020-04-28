#%%
import pandas as pd
import itertools

#%%
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
""" Classification model """

"""
def cos_sim(emb1, emb2):
    import scipy
    cos_dist = scipy.spatial.distance.cdist(emb1, emb2, "cosine")
    return cos_dist
"""

# %%

