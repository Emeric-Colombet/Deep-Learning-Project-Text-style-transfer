"""
Print train/validation/test data for reference or for modeling in GoogleColab
"""

from style_transfer.infrastructure.style_transfer_data import StyleTransferData, Subtitles
from style_transfer.domain.preprocess_data import BaseData

df = StyleTransferData.data
df_train, df_validation, df_test = BaseData(df).split_train_test(
    test_size=0.15,
    validation_size=0.2,
    random_state=42,
    text_type='combined')

path = Subtitles.find_path('data')

# Export train/validation/test data for reference
Subtitles.print_to_csv(df_train, path, 'df_train.csv')
Subtitles.print_to_csv(df_validation, path, 'df_validation.csv')
Subtitles.print_to_csv(df_test, path, 'df_test.csv')

# Export train/validation data as text file for modeling in GoogleColab
df_train_combined_only = df_train['encoded_latinamerica_spain']
df_validation_combined_only = df_validation['encoded_latinamerica_spain']

Subtitles.print_to_csv(df_train_combined_only, path, 'df_train_combined_only.txt')  # For model in GoogleColab
Subtitles.print_to_csv(df_validation_combined_only, path, 'df_validation_combined_only.txt')  # For model in GoogleColab
