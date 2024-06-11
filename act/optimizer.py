import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestOptimizer():

    def __init__(self, random_state):
        self.model = RandomForestRegressor(n_estimators = 1000, random_state = random_state)

    def fit(self, summary_features_train, g_train):
        self.model.fit(summary_features_train, g_train)

    def predict(self, summary_features_test):
        return self.model.predict(summary_features_test)
    
    def predict_proba(self, summary_features_test):
        return self.model.predict_proba(summary_features_test)