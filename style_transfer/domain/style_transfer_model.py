"""This module load our style transfer model """

from xmlrpc.client import Boolean
import os 
import pickle
import pandas as pd

class BaseStyleTransferModel:
    """
    Interface of our ChurnModel

    """
    PICKLE_ROOT = "data/models"

    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model from the training set (X, y)."""
        return "fit"

    
    def predict(self, X: pd.DataFrame):
        """Return predictions."""
        return f"This is the prediction {X}"

    @classmethod
    def load(cls) -> Boolean:
        """Load the churn model from the class instance name."""
        pass

    def save(self) -> Boolean:
        """Save the model (pkl format)."""
        pass



#TODO : Faire un modèle qui hérite de BaseStyleTransferModel, et de Base estimator de scikit-learn pour travailler avec un pipeline. 