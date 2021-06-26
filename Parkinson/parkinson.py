import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix

import time


class Classification():
    def __init__(self, filename, features, label_col, test_size=0.2):
        self.df = pd.read_csv(filename)
        self.X, self.y = self.df[features], self.df[label_col].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size)
        self.X_train_engineer = None
        self.X_test_engineer = None
        self.method_preprocessing = None
        self.method_classifier = None
        self.params_classifier = None
        self.method_set = False
        self.parameter_set = False
    
    def timing(function):
        def wrapper(self):
            print('Running ', function.__name__)
            print('Start timing at: ', time.time())
            start = time.time()
            function(self)
            stop = time.time()
            print('Time elapsed: ', stop - start)
            
        return wrapper
    
    def set_methods(self, preprocess, clf):
        self.method_preprocessing = preprocess
        self.method_classifier = clf
        self.method_set = True
        
    def set_parameters(self, params):
        self.params_classifier = params
        self.parameter_set = True
    
    @timing
    def data_preparation(self):
        print('----- Running data preprocessing. Currently nothing to do...')
        pass
    
    @timing
    def preprocessing(self):
        self.X_train_engineer = self.method_preprocessing.fit_transform(self.X_train)
        self.X_test_engineer = self.method_preprocessing.transform(self.X_test)
        
    @timing
    def classifier(self):
        if self.parameter_set is False:
            raise Exception('Error: All parameters are not yet set!')
        grid = GridSearchCV(estimator=self.method_classifier, 
                            param_grid=self.params_classifier)
        grid.fit(self.X_train_engineer, self.y_train)        
        self.method_classifier = grid.best_estimator_
        
    @timing
    def evaluate(self):
        preds = self.method_classifier.predict(self.X_test_engineer)
        if self.method_classifier.__class__.__name__ == 'XGBClassifier':
            xgb.plot_importance(self.method_classifier)
            plt.show()
        elif self.method_classifier.__class__.__name__ == 'RandomForestClassifier':
            importances = self.method_classifier.feature_importances_
            indices = np.argsort(importances)

            plt.figure(figsize=(15, 8))
            plt.barh(range(len(indices)), importances[indices])
            plt.show()
        print('accuracy_score: ', accuracy_score(self.y_test, preds))
        print('confusion matrix for ', self.method_classifier.__class__.__name__ , ": ", confusion_matrix(self.y_test, preds))
    
    @timing
    def run(self):
        if self.method_set is False:
            raise Exception('Methods are not yet set. Aborting!')
        if self.parameter_set is False:
            print('Warning: All parameters are taking default values. Consider tuning!')
        self.data_preparation()
        self.preprocessing()
        self.classifier()
        self.evaluate()