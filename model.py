# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
import pickle
import requests 
import json 

# Importing the dataset
data = pd.read_csv('data/upsampled_data.csv')

import xgboost as xgb
from sklearn.model_selection import train_test_split
x = data[['Performance','Dyspnoea','Cough','TNM','DM']]
y = data['Risk1Y']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)
classifier = xgb.sklearn.XGBClassifier(nthread=-1, seed=1)
classifier.fit(X_train, y_train)




#################################################
# Predicting the Test set results
y_pred = xgb.predict(X_test)

# Saving model to disk
pickle.dump(xgb, open('xgb_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('xgb_model.pkl','rb'))

#print(model.predict([[]]))




