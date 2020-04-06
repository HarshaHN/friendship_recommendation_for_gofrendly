"""
Date: 5 Apr 2020
Author: Harsha harshahn@kth.se
Implementation of evaluation metrics
"""
#%%
""" Import libraries """
import numpy as np

#==============================================================================

"""1. Metrics of Relevance"""
#%%
"""a. Area under ROC """
def auroc(true, score):
    from sklearn.metrics import roc_auc_score
    res = roc_auc_score(true, score)
    print("AUROC has been computed and the value is ", res)
    return res

#%%
"""b. MAP@K and MAR@K"""
# Precision@k = (# of recommended items @k that are relevant) / (# of recommended items @k)
# Recall@k = (# of recommended items @k that are relevant) / (total # of relevant items)
def meanavg(query, true, score):
    from sklearn.metrics import precision_score, recall_score
    precision = []; recall = []; res = []
    for q in query:
        precision.append(precision_score(true[q], score[q]))
        recall.append(recall_score(true[q], score[q]))

    res[0] = np.mean(precision); res[1] = np.mean(recall)
    print("MAP@K and MAR@K has been computed and the values are ", res[0], "and ", res[1])
    return res

#%%
"""c. Hit-rate"""
def hitrate(frds, rec):
    a = set(frds)
    b = set(rec)
    c = a.intersection(b) 
    res =  len(c)/len(b)  
    print("Hitrate has been computed and the value is ", res)
    return res

#==============================================================================

"""2. Metrics of Serendipity"""
#%%
"""a. Personalization """
def personalization():
    res = 0

    print("Personalization of users has been computed and the value is ", res)
    return res

#%%
"""b. Diversity """
def diversity():
    res = 0

    print("Diversity of the list has been computed and the value is ", res)
    return res
#==============================================================================


"""3. Metrics of User Hits"""
#%%
"""a. Link-up rate """
def linkuprate():
    res=0

    print("Link-up rate has been computed and the value is ", res)
    return res

#%%
"""b. User hits ratio """
def userhits():
    res=0
    
    print("User hits ratio has been computed and the value is ", res)
    return res

#==============================================================================

"""4. Rank aware metric"""
#%%
"""a. Mean Reciprocal Rank (MRR) """
def mrr(frds, rec):
    res = 0
    for i in frds:
        if i in rec:
            res += 1/(rec.index(i)+1)
    res = res/len(frds)
    print("MRR has been computed and the value is ", res)
    return res

#==============================================================================