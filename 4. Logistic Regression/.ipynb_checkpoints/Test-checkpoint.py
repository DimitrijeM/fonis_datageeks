# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 08:04:19 2018

@author: Sandro Radovanovic
"""

# Load Data
from sklearn import datasets
iris = datasets.load_iris()

# Define Data
X = iris.data[:, :4]
y = (iris.target != 0) * 1

# Train Model
model = LogisticRegression(lr = 0.1, num_iter = 1000)

# Learn Thetas 
model.fit(X, y)
model.theta

model.predict_proba(X)
model.predict(X)

# Train Ridge Logistic Regression
model = RidgeLogisticRegression(lr = 0.1, num_iter = 1000, C = 1)

# Learn Thetas
model.fit(X, y)
model.theta

# Train Lasso Logistic Regression
model = LassoLogisticRegression(lr = 0.1, num_iter = 1000, C = 2)

model.fit(X, y)
model.theta
