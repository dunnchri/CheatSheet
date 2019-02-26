# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:42:37 2019

@author: 599817
"""

## ML Cheatsheet!
#1) identify features and target. Set them to X and y
#2) perform a test train split. Use the test train split to create X_train, X_test, y_train, y_test 
#2b) **Optional based on the algorithm** scale or transform your variables
#3) Instantiate a model object (i.e. knn = K_NeighborsClassifier() )
#4) train the model using model.fit(). This is where you input your model hyperparameters (i.e. k=5)
#5) make predictions on your test set using model.predict()
#6) Measure accuracy, retune model if needed

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

## step 1
model_vars = ['al','ri']
X = glass[model_vars]
y = glass.household

## step 2
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41)

## step 2b
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## step 3
model = LogisticRegression()

## step 4
model.fit(X_train, y_train)

## step 5
y_pred_test = model.predict(X_test)

## step 6
print('train score')
print(model.score(X_train, y_train))

print('test score:')
print(model.score(X_test, y_test))