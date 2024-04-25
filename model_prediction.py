# Project 3 SVM Model

import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Read Dataset
df = pd.read_csv('cbb.csv')

# Remove rows with year 2023
df = df[df['YEAR'] != 2023]

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

# Train and fit SVM model        
svm_m = svm.SVC(kernel='rbf', C=100, probability = True, class_weight="balanced", gamma='scale', tol=0.0001)
svm_model = Pipeline ([
    ('scaler', StandardScaler()),
    ('svm', svm_m)
])
svm_model.fit(X_train, y_train)

# Accuracy of model
print("Accuracy of train: ", svm_model.score(X_train, y_train))
print("Accuracy of test : ", svm_model.score(X_test, y_test))

y_pred = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Read and convert test data
s23_df = pd.read_csv('cbb23.csv')
s23_df['POSTSEASON'] = s23_df['POSTSEASON'].map(convert)
s23_df.fillna({'POSTSEASON': 9}, inplace = True)
X_new = s23_df[['G','W','ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D','TOR','TORD','ORB','DRB','FTR','FTRD','2P_O','2P_D','3P_O',
       '3P_D','ADJ_T']]
y_new = s23_df['POSTSEASON']

# Predict POSTSEASON ranking for test data                     
svm_predictions = svm_model.predict(X_new)
svm_probs= svm_predictions.astype(int)
teams = s23_df.loc[y_new.index, 'TEAM']
svm_predictions_df = pd.DataFrame({'TEAM': teams, 'Predictions': svm_probs})
svm_predictions_df['Predictions'] = svm_predictions_df['Predictions']
svm_predictions_df.to_csv('svm_pred.csv', mode='w', index=False)

# Calculate accuracy and confusion matrix of test
accuracy_final_predictions = accuracy_score(y_new, svm_predictions)
print("\nAccuracy of 2023 predictions:", accuracy_final_predictions)
cm_final_predictions = confusion_matrix(y_new, svm_predictions)
print("Confusion Matrix for 2023 predictions:")
print(cm_final_predictions)
