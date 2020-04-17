
#%%
""" Exploratory Data Analysis """
# Perform EDA from pandas
# https://medium.com/datadriveninvestor/introduction-to-exploratory-data-analysis-682eb64063ff
# Use data pre-processing libs such as 
query = [
    "SELECT user_id, iAm, meetFor, Age, marital, children FROM user_details \
        INNER JOIN users ON user_details.user_id=users.id",
    "SELECT user_id, birthday, city, country, lat, lng FROM user_details \
        INNER JOIN users ON user_details.user_id=users.id"
    ]




#%%
""" 2. Classification model """
query = [
        """ a. Positive samples """
        # 1. Chat friends(hard positive)
        # 2. Mutually connected friends(hard positive)
        # count: a. b. 240,824 c. 
        "SELECT a.user_id, a.friend_id FROM friends a\
        INNER JOIN friends b\
        ON (a.user_id = b.friend_id) AND (a.friend_id = b.user_id)",
        # 3. Activity friends(soft positive)
        # count: a. b.32,422 c. 
        "SELECT activity_id, user_id FROM activity_invites where isGoing = 1",
        # 4. Chat friends of chat friends(soft positive)
        # Comments: 1 and 2 may overlap, 3 shows common interest & may lead to more 
        # of those, 4 can be populated however may never have seen each other.

        """b. Negative samples"""
        # 1. Blocked user pairs (hard negative) 
        # count: a. b.13,684 c. 
        "SELECT user_id, blocked_id FROM blocked_users",

        # 2. Viewed users but not added as friends (hard negative)
        # Viewed users count: a. b.4,464,793(4,846,799) c. 
        "SELECT a.user_id, a.seen_id FROM seen_users a\
        LEFT JOIN friends b\
        ON (b.user_id = a.user_id) AND (b.friend_id = a.seen_id)\
        WHERE b.user_id IS null"

        # 3. one-way added as friend but not chat friends (hard negative)
        # one-way friends count: a. b.1,102,338 c.
        #Comments: 

        #Overall comments:
        # A = A-B leading to mutually exclusive groups.
        ]


#%%
""" 3. Network aggregation """

#%% I/O data architect
""" 1. User data for features """
    """ a.  """
    "SELECT user_id, myStory, iAm, meetFor, marital, children"
    #iAmCustom, meetForCustom, #describeYourself,

    Euclidean (GeoLoc, Age, Children, ChildrenAge), 
    Binary (Marital status). Add and normalize (Homophily). 
    Train MLP for all and retrain final classifier Yes/No (maybe multiple times or overtrain)
    Use validation data to hybridize the scores i.e., to find weights.

    # 1. Select user profile data 
    "SELECT user_id, isActive, isProfileCompleted, lang, myStory, describeYourself, iAmCustom, meetForCustom, iAm, meetFor, marital, children \
        FROM user_details \
            INNER JOIN users ON user_details.user_id=users.id",
    # 2. Select friend links
    "SELECT user_id, friend_id FROM friends", 
    # 3. Activities data
    "SELECT id, title, description FROM activities",
    # 4. Activity links
    "SELECT activity_id, user_id FROM activity_invites where isGoing = 1",
    # 5. Chat links
    #Q: isActive, 

#%% Data flow architect

#%% Compatibility to data evolution

#%% backup or reference
query = [
        # 1. Select user profile data 
        "SELECT user_id, isActive, isProfileCompleted, lang, myStory, describeYourself, iAmCustom, meetForCustom, iAm, meetFor, marital, children \
            FROM user_details \
                INNER JOIN users ON user_details.user_id=users.id",
        # 2. Select friend links
        "SELECT user_id, friend_id FROM friends", 
        # 3. Activities data
        "SELECT id, title, description FROM activities",
        # 4. Activity links
        "SELECT activity_id, user_id FROM activity_invites where isGoing = 1",
        # 5. Chat links
        ]
