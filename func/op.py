import os
import pandas as pd
import time

# %%
#Text cleansing
def cleanse(text):
    import re
    import emoji
    text = text.replace("\n", " ") #remove breaks
    text = emoji.get_emoji_regexp().sub(u'',text) #remove emojis
    r = re.compile(r'([.,/#!?$%^&*;:{}=_`~()-])[.,/#!?$%^&*;:{}=_`~()-]+')
    text = r.sub(r'\1', text) #multiple punctuations
    if len(text) < 10: text = None #short texts
    return text

#Translate the text to english
def trans(in_stories):
    from googletrans import Translator
    langlist =[]; #langdist = []
    for index, row in in_stories.iterrows():
        t = Translator()
        txt = cleanse(row['myStory']) #cleanse  
        if (txt != None) and (txt != ''):
            #langdist.append(t.detect(row).lang)
            time.sleep(.5)
            try: langlist.append(t.translate(txt, dest = 'en').text)
            except: 
                print( index, '\n', txt , '\n', '###'); langlist.append(None);  break;
        else: langlist.append(None)
    return langlist

def removenull(text):
    return text[(text['story'] != '') & (~text['story'].isnull())]
 
# %%
def loadone(): #load the dfs
    os.chdir('./data/in')
    uNodes = pd.read_hdf("uNodes.h5", key='uNodes')
    #fLinks = pd.read_hdf("fLinks.h5", key='fLinks')
    #aNodes = pd.read_hdf("aNodes.h5", key='aNodes')
    #aLinks = pd.read_hdf("aLinks.h5", key='aLinks')
    #cLinks = pd.read_hdf("cLinks", key='cLinks')
    os.chdir('../..')
    return uNodes#[uNodes, fLinks, aNodes, aLinks]

# %% Clear all variables
def clearvars():
    import sys
    sys.modules[__name__].__dict__.clear()