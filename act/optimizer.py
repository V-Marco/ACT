import numpy as np
from sklearn.ensemble import RandomForestRegressor

'''
RandomForestOptimizer
A minimal wrapper for sklearn RandomForestRegressor to set defaults
Contains fit and predict methods.
'''

class RandomForestOptimizer():

    def __init__(self, n_estimators: int = 1000, random_state: int = 42, max_depth: int = None):
        self.model = RandomForestRegressor(n_estimators = n_estimators, random_state = random_state, max_depth=max_depth)


    def fit(self, summary_features_train: np.ndarray, g_train:np.ndarray) -> None:
        """
        Implements sklearn fit method 
        Parameters:
        ----------
        self
        
        summary_features_train: np.ndarray
            Features from voltage trace
        
        g_train: np.ndarray
            Matching conductance values to extracted features
            
        Returns:
        ---------
        None
        """
        g_train = [g for g in g_train if str(g) != 'nan']
        self.model.fit(summary_features_train, g_train)


    def predict(self, summary_features_test: np.ndarray) -> np.ndarray:
        """
        Implements sklearn predict method
        Parameters:
        ----------
        self
        
        summary_features_test: np.ndarray
            Features 
            
        Returns:
        ----------
        prediction: np.ndarray
            Conductance set
        """
        return self.model.predict(summary_features_test)