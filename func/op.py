# %%
#Translate the text to english
def trans(sub_stories):
    from googletrans import Translator
    translator = Translator()
    langlist =[]; #langdist = []
    for index, row in sub_stories.iterrows():
        text = row['myStory']
        if (text != None) and (text != ''):
            #langdist.append(translator.detect(row).lang)
            try: 
                langlist.append(translator.translate(text, dest = 'en').text)
            except: print( index, '\n', text , '\n', '###'); break;
        else: langlist.append(None)
    return langlist

#from matplotlib import pyplot as plt
#plt.hist(langlist)