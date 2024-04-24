# Project 3 SVM Model

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Read Dataset
df = pd.read_csv('cbb.csv')

# Convert non-numeric values in column "POSTSEASON"
convert = {"Champion" : 1, "2ND" : 2, "F4" : 3, "E8" : 4, "S16" : 5, "R32" : 6, "R64" : 7, "R68" : 8}
df['POSTSEASON'] = df['POSTSEASON'].map(convert)
df['POSTSEASON'].fillna(9, inplace=True)

# Split dataframe into testing and training data
X = df[['G','W','ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O',
       '3P_D','ADJ_T']]
y = df['POSTSEASON']
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.3, stratify=y)

# Train and fit SVM model        
svm_m = svm.SVC(kernel='rbf', C=10, probability = True, class_weight="balanced", gamma='scale', tol=0.0001)
svm_model = Pipeline ([
    ('scaler', StandardScaler()),
    ('svm', svm_m)
])
svm_model.fit(X_train, y_train)

# Accuracy of model
print("Accuracy of train: ", svm_model.score(X_train, y_train))
print("Accuracy of test : ", svm_model.score(X_test, y_test))

print("Done!")
