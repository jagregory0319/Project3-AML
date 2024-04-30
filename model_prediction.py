# Project 3 - AML
# Develop a predictive model to analyze and evaluate a dataset

import pandas as pd
from sklearn.model_selection import train_test_split

# Read and Clean Dataset
df = pd.read_csv('cbb.csv')

# Convert non-numeric values in column "POSTSEASON"
convert = {"Champion" : 1, "2ND" : 2, "F4" : 3, "E8" : 4, "S16" : 5, "R32" : 6, "R64" : 7, "R68" : 8}
df['POSTSEASON'] = df['POSTSEASON'].map(convert)
df.fillna({'POSTSEASON': 9}, inplace = True)

# Split dataframe into testing and training data
X = df[['G','W','ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O',
       '3P_D','ADJ_T']]
y = df['POSTSEASON']
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.3, stratify=y)
