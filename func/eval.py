"""
Date: 5 Apr 2020
Author: Harsha harshahn@kth.se
Implementation of evaluation metrics
"""
#%%
""" Import libraries """
import numpy as np

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
    #print("MAP@K and MAR@K has been computed and the values are ", res[0], "and ", res[1])
    return res

#%%
"""c. Hit-rate"""
def hitrate(topNpredictions,leftoutpredictions):
    hits=0
    total=0
    for leftout in leftoutpredections:
        uid=leftout[0]
        leftoutmovieid=leftout[1]
        hit=false
        for movieId ,predictedRating in topNpredictions[int(userId)]:
            if(int(movieId)==int(leftoutmovieId)):
                hit=true
        if(hit):
            hits+=1
        total+=1 
    
    return hits/total 
#==============================================================================


"""2. Metrics of Serendipity"""
#%%
"""a. Personalization """


#%%
"""b. Diversity """
def Diversity(topNPredicted, simsAlgo):
        n = 0
        total = 0
        simsMatrix = simsAlgo.compute_similarities()
        for userID in topNPredicted.keys():
            pairs = itertools.combinations(topNPredicted[userID], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                innerID1 = simsAlgo.trainset.to_inner_iid(str(movie1))
                innerID2 = simsAlgo.trainset.to_inner_iid(str(movie2))
                similarity = simsMatrix[innerID1][innerID2]
                total += similarity
                n += 1

        S = total / n
        return (1-S)
#==============================================================================


"""3. Metrics of User Hits"""
#%%
"""a. Link-up rate """


#%%
"""b. User hits ratio """
def HitRate(topNPredicted, leftOutPredictions):
        hits = 0
        total = 0

        # For each left-out rating
        for leftOut in leftOutPredictions:
            userID = leftOut[0]
            leftOutMovieID = leftOut[1]
            # Is it in the predicted top 10 for this user?
            hit = False
            for movieID, predictedRating in topNPredicted[int(userID)]:
                if (int(leftOutMovieID) == int(movieID)):
                    hit = True
                    break
            if (hit) :
                hits += 1

            total += 1

        # Compute overall precision
        return hits/total
#==============================================================================

"""4. Rank aware metric"""
#%%
"""a. Mean Reciprocal Rank (MRR) """
def AverageReciprocalHitRank(topNPredicted, leftOutPredictions):
        summation = 0
        total = 0
        # For each left-out rating
        for userID, leftOutMovieID, actualRating, estimatedRating, _ in leftOutPredictions:
            # Is it in the predicted top N for this user?
            hitRank = 0
            rank = 0
            for movieID, predictedRating in topNPredicted[int(userID)]:
                rank = rank + 1
                if (int(leftOutMovieID) == movieID):
                    hitRank = rank
                    break
            if (hitRank > 0) :
                summation += 1.0 / hitRank

            total += 1

        return summation / total
#==============================================================================
