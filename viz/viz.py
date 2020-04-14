#%%
import pandas as pd
fpairs = pd.read_hdf("../data/vars/fLinks.h5", key='fpairs')

import networkx as nx
G = nx.Graph()

# %%
G.clear()
for a in fpairs.iteritems():
    G.add_edge(*a[1])
del a

# %%
""" Viz """
#nx.write_edgelist(G, "test.edgelist")
nx.write_graphml(G, "gf.graphml") 
#nx.draw_networkx(G,  with_labels=True)

#%%
import matplotlib.pyplot as plt
import networkx as nx



# %%
