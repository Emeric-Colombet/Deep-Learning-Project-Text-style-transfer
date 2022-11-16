"""This module loads the preprocessing object """
from sklearn.base import BaseEstimator, TransformerMixin


class PreprocessData(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
            
    def fit(self, df):
        return "fit"

    def transform(self, df):
        raise NotImplementedError
        df = df.copy()
        df["col2"] = df["col1"].apply(self.tokenization, args=("tokenize_arg"))  
        df["col3"] = df["col1"].apply(self.lemmatization, args=("lemmatization_arg"))  
        df["col4"] = df["col1"].apply(self.lowercase, args=("lowercase_arg"))  
        df["col5"] = df["col1"].apply(self.stopwords, args=("stopwords_arg"))  
        

    @staticmethod
    def tokenization(tokenize_arg):
        pass

    @staticmethod
    def lemmatization(lemmatization_arg):
        pass
    
    @staticmethod
    def lowercase(lowercase_arg):
        pass
    
    @staticmethod
    def stopwords(stopwords_arg):
        pass