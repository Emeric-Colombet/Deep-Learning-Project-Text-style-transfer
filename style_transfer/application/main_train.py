import logging
from style_transfer.infrastructure.style_transfer_data import StyleTransferData
from style_transfer.domain.style_transfer_model import BaseStyleTransferModel, TransformerStyleTransferModel
from style_transfer.domain.preprocess_data import PreprocessData
from style_transfer.infrastructure.style_transfer_data import StyleTransferData
from style_transfer.domain.preprocess_data import EuropeanSpanishTerms, BaseData
from evaluate.evaluator import Text2TextGenerationEvaluator
df = StyleTransferData.data
df = EuropeanSpanishTerms(df).count_regional_terms()
df_train, df_validation, df_test = BaseData(df).split_train_test(test_size=0.15, validation_size=0.2, random_state=42)
#TODO : Enlever ces deux lignes là et les mettre dans un preprocessing. 
df_train["combined"] = '<s>' + df_train["text_latinamerica"] + '</s>' + '>>>>' + '<p>' + df_train["text_spain"] + '</p>'
df_validation["combined"] = '<s>' + df_validation["text_latinamerica"] + '</s>' + '>>>>' + '<p>' + df_validation["text_spain"] + '</p>'
logging.info("Data loaded")
gpt = TransformerStyleTransferModel(1,model_name="Latino_to_European",tokenizer_name='DeepESP/gpt2-spanish',batch_size=8,cache_dir='cache',output_dir='Latino_to_European')
gpt.fit(df_train,df_validation)
logging.info("Model fitted")
