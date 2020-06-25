

#%%--------------------
""" 03. GCP translator API """

class gcpserver:

    def gcptrans(text):
        #Corpus with example sentences
        """
        texts = [ 'A man is eating food.',
                    'A man is eating a piece of bread.',
                    'The girl is carrying a baby.',
                    'A man is riding a horse.',
                    'A woman is playing violin.',
                    'Two men pushed carts through the woods.',
                    'A man is riding a white horse on an enclosed ground.',
                    'A monkey is playing drums.',
                    'A cheetah is running behind its prey.']
        #import random; res = texts[random.randint(0,8)]
        """
        import numpy as np
        return np.random.randn(1204) 

#%%-------------------
""" 02. stories translation with GCP """

import pandas as pd
df = pd.read_hdf("../data/one/trainfeat.h5", key='01')
df = df.drop(columns=['iam', 'meetfor', 'age', 'marital', 'kids', 'lat', 'lng'])
num_chars = df.story.apply(lambda x: len(x) if x!=-1 else 0).sum(0) #3,947,808
cost = num_chars*20/1000000-10; 
print('cost =', round(cost),'USD or', round(cost*9.33),'SEK')
# cost = 69.0 USD or 643.0 SEK

#df['story'] = df['story'].apply(lambda x: gcpserver.gcptrans(x) if x!=-1 else -1)
# df['story'].to_hdf("../data/one/stories.h5", key='01')


#%%---------------------
""" SBERT """

# df = pd.read_hdf("../data/one/stories.h5", key='01')
from sentence_transformers import SentenceTransformer
sbertmodel = SentenceTransformer('roberta-large-nli-mean-tokens')

# Generate embeddings
before = time.time() #listup = lambda x: [x]
df['emb'] = df['story'].apply(lambda x: sbertmodel.encode([x]) if x!=-1 else -1)
print("-> S-BERT embedding finished.", (time.time() - before)) #534 sec
#df.drop(columns = 'story', inplace = True)
# df.to_hdf("../data/one/emb.h5", key='01')
