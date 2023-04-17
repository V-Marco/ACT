import os, sys, importlib, csv

import numpy as np
import pandas as pd



class AutomaticCellTuner:

    def __init__(self) -> None:

        self.feature_model = None
        self.optimizer = None

    def fit_predict(self):

        # Checks

        if self.feature_model is None:
            raise RuntimeError("Feature model is not set.")
        
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not set.")
        


    

