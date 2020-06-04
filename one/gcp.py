

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

        # df = pd.read_hdf("../data/one/feat.h5", key='01')
        from gcp import gcpserver
        df['story'] = df['story'].apply(lambda x: gcpserver.gcptrans(x) if x!=-1 else -1)
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