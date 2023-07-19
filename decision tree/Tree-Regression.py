# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 14:57:36 2023

@author: sara
"""

# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 2:3].values
y = dataset.iloc[:, 4].values


# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#sc_y = StandardScaler()
#X_train = sc_X.fit_transform(X_train) 
#X_test = sc_X.fit_transform(X_test) 



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
total_acc = 0
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
    # Training the Decision Tree Regression model on the whole dataset
    from sklearn.tree import DecisionTreeClassifier
    classifire = DecisionTreeClassifier(criterion= "entropy")
    classifire.fit(X_train, y_train)
    
    
    # Predicting the Test set results
    y_pred = classifire.predict(X_test)
    
    #Making the confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    acc =(cm[0][0]+cm[1][1])/len(y_test)
    print('the accuracy of logistic regression classifier is ', acc*100, '%')
    total_acc = total_acc+acc
    
final =total_acc/20
print('averaga of accuracy of 20 times  ', (final*100), '%')



