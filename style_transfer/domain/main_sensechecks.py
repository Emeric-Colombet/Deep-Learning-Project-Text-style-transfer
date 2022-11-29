"""
Apply a sense check accounting for known European Spanish terms in the prediction
To be used if a substantial list of predictions is available
"""

from style_transfer.infrastructure.style_transfer_data import StyleTransferData
from style_transfer.domain.preprocess_data import EuropeanSpanishTerms, BaseData
import pandas as pd

df = StyleTransferData.data
df_train, df_validation, df_test = BaseData(df).split_train_test(
    test_size=0.15,
    validation_size=0.2,
    random_state=42,
    text_type='combined')

# TODO: load real predictions for df_train, df_validation and df_test
prediction_train = df_train['text_spain']
prediction_validation = df_validation['text_latinamerica']
prediction_test = df_test['text_spain']

sense_score_train = EuropeanSpanishTerms(df_train).calculate_sense_score(prediction_train)
sense_score_validation = EuropeanSpanishTerms(df_validation).calculate_sense_score(prediction_validation)
sense_score_test = EuropeanSpanishTerms(df_test).calculate_sense_score(prediction_test)

sense_scores = pd.DataFrame({
    'df_train': pd.Series(sense_score_train),
    'df_validation': pd.Series(sense_score_validation),
    'df_test': pd.Series(sense_score_test)})

print(sense_scores)
