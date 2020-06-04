#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
""" Exploratory Data Analysis 
# Perform EDA from pandas
# https://medium.com/datadriveninvestor/introduction-to-exploratory-data-analysis-682eb64063ff
https://towardsdatascience.com/a-starter-pack-to-exploratory-data-analysis-with-python-pandas-seaborn-and-scikit-learn-a77889485baf


'in' : "SELECT user_id, iAm, meetFor, birthday, marital, children FROM user_details \
INNER JOIN users ON user_details.user_id=users.id",
main =  pd.read_hdf("./data/raw/in.h5", key='in')

'info' : "SELECT user_id, birthday, city, country, lat, lng FROM user_details 
main =  pd.read_hdf("./data/raw/in.h5", key='info')

Göteborg
user vectors = [ sbert_emb, values(iam, meetFor, marital, has children), eucld_diff(age, lat, lng) in range(0,1) ]
in_vec to c-model = [ cosine_sim_sbert, count(intersection(iam, meetFor)/3) binary_equality(marital, has children), eucld_diff(age, lat, lng) in range(0,1) ]

Because don't have lot of data which mlp/dnn usually needs for such large input vector. conceptually/visually this translates to '1' for the intersection regions, '0' for non and '0.5' for the defaults.
variants: interlay of raw 1024 vectors, 
"""

#%%
main =  pd.read_hdf("./data/raw/in.h5", key='info')
data = main.head(100)

data.dropna(inplace=True)
data.info()
data.describe()
data.head(5)

main.dropna(inplace=True)
main.info()
main.describe()
main.head(5)

main['country'].value_counts()
main['city'].value_counts()

sns.set(style="darkgrid")
ax = sns.countplot(y="country", data=main)


#Numerical attributes
num = df.select_dtypes(include=['int'])
num.hist()

#Categorical attributes
sns.countplot(data=df.select_dtypes(include=['object'], x ='iAm')


# %%
