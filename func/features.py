#%%-------------
""" Cleanse myStory """
import re
import emoji #conda install -c conda-forge emoji
class cleanse:
    #cleanse myStory
    r = re.compile(r'([.,/#!?$%^&*;:{}=_`~()-])[.,/#!?$%^&*;:{}=_`~()-]+')

    @classmethod
    def cleanse(cls, text):
        if (text == '') or (text == None): #.isnull()
            text = -1
        else:
            text = text.replace("\n", ". ") #remove breaks
            text = emoji.get_emoji_regexp().sub(u'',text) #remove emojis
            text = cls.r.sub(r'\1', text) #multiple punctuations
            if len(text) < 10: 
                text = -1 #short texts
        return text

#%%--------------------
""" GCP translator API """
