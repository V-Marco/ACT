import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestOptimizer():

    def __init__(self, n_estimators=1000, random_state=42):
        self.model = RandomForestRegressor(n_estimators = n_estimators, random_state = random_state)

    def fit(self, summary_features_train, g_train):
        g_train = [g for g in g_train if str(g) != 'nan']
        self.model.fit(summary_features_train, g_train)

    def predict(self, summary_features_test):
        return self.model.predict(summary_features_test)