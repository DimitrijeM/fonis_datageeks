# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Librarires
import numpy as np

class LogisticRegression:
    def __init__(self, lr = 0.01, num_iter = 100, fit_intercept = True, verbose = False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis = 1)
    
    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        # Initial weight initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            if(self.verbose == True and i % 100 == 0):
                print(f'Logistic Loss: {self.__loss(h, y)}')
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
                
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold = 0.5):
        return self.predict_proba(X) >= threshold
    
class RidgeLogisticRegression:
    def __init__(self, lr = 0.01, C = 0.1, num_iter = 100, fit_intercept = True, verbose = False):
        self.lr = lr
        self.C = C
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis = 1)
    
    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def __loss(self, h, y):
	reg = 0
        if self.fit_intercept:
            reg += self.C/2 * np.sum(np.power(self.theta[1:], 2))
        else:
            reg += self.C/2 * np.sum(np.power(self.theta, 2))
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + reg
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        # Initial weight initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            reg = 0
            if self.fit_intercept:
                reg += self.C * np.append(0, self.theta[1:])
            else:
                reg += self.C * self.theta
            if(self.verbose == True and i % 100 == 0):
                print(f'Logistic Loss: {self.__loss(h, y)}')
            gradient = np.dot(X.T, (h - y)) / y.size + reg
            self.theta -= self.lr * gradient
                
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold = 0.5):
        return self.predict_proba(X) >= threshold
    
class LassoLogisticRegression:
    def __init__(self, lr = 0.01, C = 0.1, num_iter = 100, fit_intercept = True, verbose = False):
        self.lr = lr
        self.C = C
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis = 1)
    
    def __sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def __loss(self, h, y):
	reg = 0
        if self.fit_intercept:
            reg += self.C * np.sum(np.abs(self.theta[1:]))
        else:
            reg += self.C * np.sum(np.abs(self.theta))
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean() + reg
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
            
        # Initial weight initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            reg = 0
            if self.fit_intercept:
                reg += self.C * np.append(0, np.sign(self.theta[1:]))
            else:
                reg += self.C * np.sign(self.theta)
            if(self.verbose == True and i % 100 == 0):
                print(f'Logistic Loss: {self.__loss(h, y)}')
            gradient = np.dot(X.T, (h - y)) / y.size + reg
            self.theta -= self.lr * gradient
                
    def predict_proba(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold = 0.5):
        return self.predict_proba(X) >= threshold