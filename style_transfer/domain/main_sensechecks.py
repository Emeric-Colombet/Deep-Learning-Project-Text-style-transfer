"""
Apply a sense check accounting for known European Spanish terms in the prediction
To be used if a substantial list of predictions is available
"""
from style_transfer.infrastructure.style_transfer_data import Subtitles
from style_transfer.domain.preprocess_data import EuropeanSpanishTerms
import pandas as pd

path = Subtitles.find_path('models/output_model.csv')
predicted_data = pd.read_csv(path)
predictions = predicted_data['text_prediction']
data = predicted_data.drop(['text_prediction'], axis=1)

sense_score = EuropeanSpanishTerms(data).calculate_sense_score(predictions)
print(sense_score)
