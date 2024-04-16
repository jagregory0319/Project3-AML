# Project 3 - AML
# Develop a predictive model to analyze and evaluate a dataset

import pandas as pd
# Read and Clean Dataset
df = pd.read_csv('cbb.csv')
df = df.dropna()
#Convert non-numeric values in column "POSTSEASON"
convert = {"Champion" : 1, "2ND" : 2, "F4" : 3, "E8" : 4, "S16" : 5, "R32" : 6, "R64" : 7, "R68" : 8}
