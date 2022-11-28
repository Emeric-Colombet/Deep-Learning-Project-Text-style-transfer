"""This module loads the preprocessing object """
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import numpy as np
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


class EuropeanSpanishTerms:
    """Class that flags most popular European Spanish terms"""

    def __init__(self, df: pd.DataFrame):
        """Constructor to initialize EuropeanSpanishTerms class

        :parameters:
            df: DataFrame with equivalent subtitle content
        """

        self.df = df

    def count_regional_terms(self) -> pd.DataFrame:
        """
        Count number of European Spanish terms for each phrase

        :returns:
            df: DataFrame with European Spanish terms count
        """

        WORDS_TARGET = [
            # Exclamations
            r'\bguay\b', r'\bguai\b', r'\benhorabuena\b', r'\bmadre mía\b', r'\bhostia', r'\bjod',
            r'\bestupendo\b', r'\bcoñ', r'\bputada', r'\bjol[i,í]n', r'\bnarices', r'\bhala\b', r'\byo qué sé\b',
            r'\bla suda'

            # Pronouns
            r'\bguarra\b', r'\bgilipolla', r'\bbuenorra\b', r'\bputa\b', r'\bzorra\b', r'\bnena\b',
            r'\btí[a,o][s]{0,1}\b', r'\bchaval', r'\bcotill', r'\bcapullo', r'\bcrack',

            # Nouns
            r'\bmóvil', r'\bcoche', r'\baparca', r'\bcamarer', r'\bcaña', r'\bpiso', r'\bpolla',
            r'\bservicios\b', r'\btorti', r'\bpolvo\b', r'\bleche\b', r'\brollo', r'\bporr', r'\bchasco',
            r'váter', r'\bpasta\b', r'\bpóliza', r'coj[o,ó]n', r'\blí[o,a]', r'\bfollón', r'\bcacahuete',
            r'\bloro', r'\bcoco', r'\bmorro', r'\bplantón', r'\blavabo', r'\bsujetador', r'\bmaletero',
            r'\bfontanero', r'\bzumo', r'\bcuerno', r'\btorta', r'\bcalcet', r'\bbraga', r'\bgafa', r'\bbañera',
            r'\bgrifo', r'\bfrigo', r'\bordenador', r'\bcerilla', r'\bprisa',

            # Adjectives
            r'\bmenudo\b', r'\bmona', r'\bmono\b', r'\bputo\b', r'\bguap', r'\bfatal\b', r'\bnato\b',
            r'\bmogoll', r'\bcurra[n]{0,1}d', r'\blince', r'\bcutre'

            # Verbs
            r'\bapetec', r'\bpilla', r'\bapañ', r'\bmenear\b', r'\bmola', r'\bliga', r'\bflip', r'\bfoll',
            r'\blia', r'\brayan', r'\bpilla'

            # Spanish sayings
            r'\ba por\b', r'\bvenga\b', r'\bvale\b', r'\bperdona\b', r'\bno pasa nada\b', r'\banda\b',
            r'\btela\b', r'\bpor saco\b'

            # Spanish conjugations
            r'\bos\b', r'aos\b', r'áos\b', r'ais\b', r'áis\b', r'eis\b', r'éis\b', r'idme\b', r'adme\b',
            r'ead\b', r'\bvuestr',

            # Less known or less important
            r'\btres pueblos', r'\bfre(.*) espárrago', r'\bmosca', r'\bplanchar la oreja', r'\bsujeatavela'
            r'\bcomer[a-z]{0,1}[e]{0,1} el tarro',
            r'\bplomo', r'\bmorad', r'\ben vela\b', r'\bla pinza'
        ]

        self.df['terms_spain_nb'] = 0

        for w in WORDS_TARGET:
            self.df['terms_spain_nb'] = np.where(
                (self.df['text_spain'].str.contains(w)) & ~(self.df['text_latinamerica'].str.contains(w)),
                self.df['terms_spain_nb'] + 1,
                self.df['terms_spain_nb'])

        self.df['terms_spain_flag'] = np.sign(self.df['terms_spain_nb'])

        return self.df


class BaseData:
    """Class that sets up base data for preprocessing"""

    def __init__(self, df: pd.DataFrame):
        """Constructor to initialize BaseData class

        :parameters:
            df: DataFrame with equivalent subtitle content
        """

        self.df = df

    @classmethod
    def format_df_for_model(cls, df: pd.DataFrame, text_type=None) -> pd.DataFrame:
        """Combines original and target texts if model input needs it to be that way

        :parameters:
            df: DataFrame containing original and target texts

        :returns:
            text_type: 'combined', 'encoded' or None, states whether the texts should be combined or not. If encoded, 
            the function add tags <s> and <p>.
        """
        if text_type == 'combined':

            df['encoded_latinamerica_spain'] = '<s>' + df['text_latinamerica'] + '</s>' + \
                             '>>>>' + \
                             '<p>' + df['text_spain'] + '</p>'

            df['encoded_latinamerica'] = '<s>' + df['text_latinamerica'] + '</s>' + \
                                         '>>>>' + \
                                         '<p>'
        if text_type == 'encoded':

            df['encoded_latinamerica'] = '<s>' + df['text_latinamerica'] + '</s>' + \
                                         '>>>>' + \
                                         '<p>'


        return df

    def split_train_test(self, test_size=0.15, validation_size=0.20, random_state=42, text_type=None):
        """Splits DataFrame into different train, validation and test subsets

        :parameters:
            test_size: Size of test data out of test and train+validation sets, 15% by default
            validation_size: Size of validation data out of validation and train sets (excluding test), 20% by default
            random_state: Seed for random state, 42 by default
            text_type: 'combined' or None, states whether the texts should be combined or not

        :returns:
            df_train: Dataframe to be used to train model
            df_validation: Dataframe to be used for validation
            df_test: Dataframe to be used for test
        """

        COLUMNS_TO_DROP = ['start_time_range', 'title', 'episode', 'terms_spain_nb', 'terms_spain_flag']
        COLUMNS_TO_DROP_LAST = ['title_terms']

        df = EuropeanSpanishTerms(self.df).count_regional_terms()
        df = self.format_df_for_model(df, text_type=text_type)

        """
        "y-label" based on title AND whether European Spanish terms were part of the target phrase
        in order to have test/train sets that have a similar distribution of shows/films (balanced way of speaking)
        and to have a similar amount of phrases where we are sure to have a regional difference between texts
        the "y-label" is not used for prediction necessarily but helps us distribute the data across data sets
        """
        df['title_terms'] = df['title'] + '-' + df['terms_spain_flag'].astype('string')

        df = df.drop(COLUMNS_TO_DROP, axis=1)
        df = df.drop_duplicates()

        # Split Test vs (Train/Validation) sets
        y = df['title_terms']
        stratified_split = StratifiedShuffleSplit(test_size=test_size, random_state=random_state)
        for train_index, test_index in stratified_split.split(df, y):
            df_train_val, df_test = df.iloc[train_index].copy(), df.iloc[test_index].copy()

        # Split Train vs Validation sets
        y_train_val = df_train_val['title_terms']
        stratified_split = StratifiedShuffleSplit(test_size=validation_size, random_state=random_state)
        for train_index, test_index in stratified_split.split(df_train_val, y_train_val):
            df_train, df_validation = df_train_val.iloc[train_index].copy(), df_train_val.iloc[test_index].copy()

        # For exploration purposes
        df_train_count = df_train.groupby(['title_terms']).count()
        df_validation_count = df_validation.groupby(['title_terms']).count()
        df_test_count = df_test.groupby(['title_terms']).count()

        for d in [df_train, df_validation, df_test]:
            d = d.drop(COLUMNS_TO_DROP_LAST, axis=1, inplace=True)
            # d = d.drop_duplicates(inplace=True)  # Not working, not sure why

        return df_train, df_validation, df_test
    @staticmethod
    def utils_decode_model_output(text: list) -> list :
        """ This function return a cleaned version of the output of the model prediction. 
        - Remove the character `>>>><p>`
        - Keep just the first proposition of the model because sometimes, 
            the Transformers generate a second sentence with the <s> and <p> tags.
        """
        decoded_text = []
        for sentence in text :
            extracted_output = sentence.split('>>>><p>')
            first_response = extracted_output[1].split('</p>')[0]
            decoded_text.append(first_response)
        return decoded_text
    @staticmethod
    def utils_from_str_to_pandas(list_char : str) -> pd.DataFrame :
        """ This function take a big list of sentences where each sentence ends with ';' 

        :parameters :
            list_char : List of all our sentences in only one string. 
    
        :returns :
            df_to_predict : A Dataframe who need to be encoded, and then feed into the model to predict

        """
        list_of_sentences = list_char.split(sep=";")
        data = {
            "text_latinamerica" : list_of_sentences
        }
        df_to_predict = pd.DataFrame(data=data)

        return df_to_predict 

