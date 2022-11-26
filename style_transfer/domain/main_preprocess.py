from style_transfer.infrastructure.style_transfer_data import StyleTransferData, Subtitles
from style_transfer.domain.preprocess_data import BaseData

df = StyleTransferData.data
df_train, df_validation, df_test = BaseData(df).split_train_test(
    test_size=0.15,
    validation_size=0.2,
    random_state=42,
    text_type='combined')

df_train_combined_only = df_train['text_latinamerica_spain']

path = Subtitles.find_path('data')
Subtitles.print_to_csv(df_train, path, 'df_train.csv')
Subtitles.print_to_csv(df_train_combined_only, path, 'df_train_combined_only.txt')  # For model in GoogleColab
Subtitles.print_to_csv(df_validation, path, 'df_validation.csv')
Subtitles.print_to_csv(df_test, path, 'df_test.csv')
