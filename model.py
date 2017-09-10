from __future__ import division
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from utils import *
import cPickle as pickle



class MyModel():
    """
    Customized random forest model for fraud prediction
    """

    def self__init__(self):
        self.cost_matrix = make_scorer(build_cost_matrix)
        self.score_func = None

    def fit(self, X, y):
        # train_test_split and fit on RF model
        self.X_train, self.X_test, self.y_train, self.y_test = \
                        train_test_split(X, y, test_size=0.33, random_state=1)
        rf = RandomForestClassifier()
        self.model = rf.fit(self.X_train, self.y_train)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def predict_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(X)

    def score(self, X_test, y_test, score_func=None):
        if score_func:
            return score_func(self.model.predict(self.X_test), self.y_test)
        else:
            return self.model.score(self.X_test, self.y_test)


def get_data(datafile, label=True):
    """
    Function for loading data and preprocessing
    """
    
    df = pd.read_json(datafile)
    if label:
        X, y = preprocess(df, label)
        return X, y
    else:
        return preprocess(df, label)

if __name__ == "__main__":
    # Loads the data and preprocesses it to fit into our model
    X, y = get_data('../files/data.json')
    model = MyModel()
    # Fits the model and predicts on the data
    model.fit(X, y)
    with open('model.pkl', 'w') as f:
        # Saves the model as a pickle file to use
        pickle.dump(model, f)
