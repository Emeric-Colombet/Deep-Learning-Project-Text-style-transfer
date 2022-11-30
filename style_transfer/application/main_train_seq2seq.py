from style_transfer.infrastructure.style_transfer_data import StyleTransferData
from style_transfer.domain.preprocess_data import BaseData
from style_transfer.domain.style_transfer_model import Seq2seqStyleTransferModel

# /!\ DO NOT RUN THIS ON BASIC CPU /!\ BEWARE /!\
# 1 EPOCH will take estimatedly 22:57:54 hours

df = StyleTransferData.data
df_cleaned, df_train, df_validation, df_test = BaseData(df).strip_accent_punctuation()

print(df_cleaned.shape)

Seq2seq_model = Seq2seqStyleTransferModel()
model, X_train, X_train_dec,\
        X_valid, X_valid_dec,\
        y_train, y_valid = Seq2seq_model.model_architecture(df_cleaned, df_train,
                                                            df_validation)

Seq2seq_model.fit(model, X_train,\
                    X_train_dec, X_valid,\
                    X_valid_dec, y_train, y_valid)