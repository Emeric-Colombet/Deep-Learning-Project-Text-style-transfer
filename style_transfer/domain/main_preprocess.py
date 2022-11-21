from style_transfer.infrastructure.style_transfer_data import StyleTransferData
from style_transfer.domain.preprocess_data import EuropeanSpanishTerms, BaseData

df = StyleTransferData.data
df = EuropeanSpanishTerms(df).count_regional_terms()
df_train, df_validation, df_test = BaseData(df).split_train_test()
