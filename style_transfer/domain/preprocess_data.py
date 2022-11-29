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

    def count_regional_terms(self, on_prediction: bool = False, word_group: str = None) -> pd.DataFrame:
        """
        Count number of European Spanish terms for each phrase

        :parameters:
            on_prediction:
                if True, compares predicted text to original text
                if False, compares Spain text to Latinamerica text

            word_group:
                if 'latest', the terms will be based on the latest list of terms to be searched
                else, the terms will be based on an initial list of terms
                since the train/validation/test split keeps a similar ratio of phrases with regional terms,
                in order to keep the same rows in the different datasets, the word list cannot be updated

        :returns:
            df: DataFrame with European Spanish terms count
        """

        WORDS_TARGET_INITIAL = [
            # Exclamations
            r'\bguay\b', r'\bguai\b', r'\benhorabuena\b', r'\bmadre mía\b', r'\bhostia', r'\bjod',
            r'\bestupendo\b', r'\bcoñ', r'\bputada', r'\bjol[i,í]n', r'\bnarices', r'\bhala\b', r'\byo qué sé\b',
            r'\bla suda'

            # Spanish sayings
            r'\ba por\b', r'\bvenga\b', r'\bvale\b', r'\bperdona\b', r'\bno pasa nada\b', r'\banda\b',
            r'\btela\b', r'\bpor saco\b'

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
            r'\bmenudo\b', r'\bmona', r'\bmono', r'\bputo', r'\bguap', r'\bfatal', r'\bnato\b',
            r'\bmogoll', r'\bcurra[n]{0,1}d', r'\blince', r'\bcutre', r'\bcachond'

            # Verbs
            r'\bapetec', r'\bpilla', r'\bapañ', r'\bmenear\b', r'\bmola', r'\bliga', r'\bflip', r'\bfoll',
            r'\blia', r'\brayan', r'\bpilla'

            # Spanish conjugations
            r'\bos\b', r'aos\b', r'áos\b', r'ais\b', r'áis\b', r'eis\b', r'éis\b', r'idme\b', r'adme\b',
            r'ead\b', r'\bvuestr',

            # Less known or less important
            r'\btres pueblos', r'\bfre(.*) espárrago', r'\bmosca', r'\bplanchar la oreja', r'\bsujeatavela'
            r'\bcomer[a-z]{0,1}[e]{0,1} el tarro', r'\bplomo', r'\bmorad', r'\ben vela\b', r'\bla pinza'
        ]

        WORDS_TARGET_LATEST = [
            # Exclamations
            r'\bguay\b', r'\bguai\b', r'\benhorabuena\b', r'\bmadre mía\b', r'\bhostia', r'\bjod',
            r'\bestupendo\b', r'\bcoñ', r'\bputada', r'\bjol[i,í]n', r'\bnarices', r'\bhala\b', r'\byo qué sé\b',
            r'\bla suda'

            # Spanish sayings
            r'\ba por\b', r'\bvenga\b', r'\bvale\b', r'\bperdona\b', r'\bno pasa nada\b', r'\banda\b',
            r'\btela\b', r'\bpor saco\b', r'\bda igual\b',

            # Pronouns
            r'\bguarra\b', r'\bgilipolla', r'\bbuenorra\b', r'\bputa\b', r'\bzorra\b', r'\bnena\b',
            r'\btí[a,o][s]{0,1}\b', r'\bchaval', r'\bcotill', r'\bcapullo', r'\bcrack',

            # Nouns
            r'\bmóvil', r'\bcoche', r'\baparca', r'\bcamarer', r'\bcaña', r'\bpiso', r'\bpolla',
            r'\bservicios\b', r'\btorti', r'\bpolvo\b', r'\bleche\b', r'\brollo', r'\bporr', r'\bchasco',
            r'váter', r'\bpasta\b', r'\bpóliza', r'coj[o,ó]n', r'\blí[o,a]', r'\bfollón', r'\bcacahuete',
            r'\bloro', r'\bcoco', r'\bmorro', r'\bplantón', r'\blavabo', r'\bsujetador', r'\bmaletero',
            r'\bfontanero', r'\bzumo', r'\bcuerno', r'\btorta', r'\bcalcet', r'\bbraga', r'\bgafa', r'\bbañera',
            r'\bgrifo', r'\bfrigo', r'\bordenador', r'\bcerilla', r'\bprisa', r'\bchupito', r'\brabo', r'\bguateque',

            # Adjectives
            r'\bmenudo\b', r'\bmona', r'\bmono', r'\bputo', r'\bguap', r'\bfatal', r'\bnato\b',
            r'\bmogoll', r'\bcurra[n]{0,1}d', r'\blince', r'\bcutre', r'\bcachond', r'\bmaj', r'\bchung',
            r'\benfarolad',

            # Verbs
            r'\bapetec', r'\bpilla', r'\bapañ', r'\bmenear\b', r'\bmola', r'\bliga', r'\bflip', r'\bfoll',
            r'\blia', r'\brayan', r'\bpilla', r'\bquedar\b', r'\bpelar\b',

            # Spanish conjugations
            r'\bos\b', r'aos\b', r'áos\b', r'ais\b', r'áis\b', r'eis\b', r'éis\b', r'idme\b', r'adme\b',
            r'ead\b', r'\bvuestr',

            # Less known or less important
            r'\btres pueblos', r'\bfre(.*) espárrago', r'\bmosca', r'\bplanchar la oreja', r'\bsujeatavela'
            r'\bcomer[a-z]{0,1}[e]{0,1} el tarro', r'\bplomo', r'\bmorad', r'\ben vela\b', r'\bla pinza'
        ]

        df = self.df.copy()

        if on_prediction:
            text_comparison = 'text_prediction'
        else:
            text_comparison = 'text_spain'

        df['terms_spain_nb'] = 0

        if word_group == 'latest':
            words = WORDS_TARGET_LATEST
        else:
            words = WORDS_TARGET_INITIAL

        for w in words:
            df['terms_spain_nb'] = np.where(
                (df[text_comparison].str.contains(w)) & ~(df['text_latinamerica'].str.contains(w)),
                df['terms_spain_nb'] + 1,
                df['terms_spain_nb'])

        df['terms_spain_flag'] = np.sign(df['terms_spain_nb'])

        return df

    def compare_predicted_regional_terms(self, prediction: pd.Series):
        """
        Counts number of European Spanish terms for the real European Spanish phrase and the prediction

        :parameters:
            prediction: Series with the prediction for text_latinamerica

        :returns:
            df: Dataframe with known regional terms (real or predicted)
        """

        self.df['text_prediction'] = prediction

        df_real = self.count_regional_terms()
        df_predicted = self.count_regional_terms(on_prediction=True, word_group='latest')
        df = self.df.copy()

        df[['terms_spain_nb', 'terms_spain_flag']] = \
            df_real[['terms_spain_nb', 'terms_spain_flag']]
        df[['terms_spain_nb_predicted', 'terms_spain_flag_predicted']] = \
            df_predicted[['terms_spain_nb', 'terms_spain_flag']]

        return df

    def calculate_sense_score(self, prediction: pd.Series) -> dict:
        """
        Sense checks if phrases with a regional difference between the Latinamerica and Spain texts
        had a Spain term included in the prediction

        :parameters:
            prediction: Series with the prediction for text_latinamerica

        :returns:
            score: Dictionary with various score values
            - regional_count:
                out of all rows WITH a regional difference,
                the share of rows with at least one regional term in the prediction

            - regional_terms:
                out of all rows WITH a regional difference, for all regional terms found in all rows,
                the share of regional terms in the prediction

            - nonregional_count:
                out of all the rows with NO evident regional difference,
                how many rows had a regional term show up in the prediction
        """

        df = self.compare_predicted_regional_terms(prediction)

        expected_rows_regional = np.where(df['terms_spain_flag'] == 1, df['terms_spain_flag'], 0).sum()
        expected_terms_regional = np.where(df['terms_spain_flag'] == 1, df['terms_spain_nb'], 0).sum()

        predicted_rows_regional = np.where(df['terms_spain_flag'] == 1, df['terms_spain_flag_predicted'], 0).sum()
        predicted_terms_regional = np.where(df['terms_spain_flag'] == 1, df['terms_spain_nb_predicted'], 0).sum()

        expected_rows_nonregional = np.where(df['terms_spain_flag'] == 0, 1, 0).sum()
        predicted_rows_nonregional = np.where(df['terms_spain_flag'] == 0, df['terms_spain_flag_predicted'], 0).sum()

        score = {
            'regional_count': predicted_rows_regional * 1.0 / expected_rows_regional,
            'regional_terms': predicted_terms_regional * 1.0 / expected_terms_regional,
            'nonregional_count': predicted_rows_nonregional * 1.0 / expected_rows_nonregional
        }

        return score


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
        if text_type == 'encoded' or text_type == 'combined':

            df['encoded_latinamerica'] = '<s>' + df['text_latinamerica'] + '</s>' + \
                                         '>>>>' + \
                                         '<p>'
        if text_type == 'combined':

            df['encoded_latinamerica_spain'] = '<s>' + df['text_latinamerica'] + '</s>' + \
                             '>>>>' + \
                             '<p>' + df['text_spain'] + '</p>'

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
    def utils_decode_model_output(text: list) -> list:
        """ This function return a cleaned version of the output of the model prediction. 
        - Remove the character `>>>><p>`
        - Keep just the first proposition of the model because sometimes, 
            the Transformers generate a second sentence with the <s> and <p> tags.
        """
        decoded_text = []
        for sentence in text:
            extracted_output = sentence.split('>>>><p>')
            first_response = extracted_output[1].split('</p>')[0]
            decoded_text.append(first_response)
        return decoded_text

    @staticmethod
    def utils_from_str_to_pandas(list_char: str) -> pd.DataFrame:
        """ This function take a big list of sentences where each sentence ends with ';' 

        :parameters:
            list_char : List of all our sentences in only one string. 
    
        :returns:
            df_to_predict : A Dataframe who need to be encoded, and then feed into the model to predict

        """
        list_of_sentences = list_char.split(sep=";")
        data = {
            "text_latinamerica": list_of_sentences
        }
        df_to_predict = pd.DataFrame(data=data)

        return df_to_predict 
