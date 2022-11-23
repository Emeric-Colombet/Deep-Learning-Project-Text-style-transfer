import logging

from style_transfer.infrastructure.style_transfer_data import StyleTransferData
from style_transfer.domain.preprocess_data import BaseData
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel

# Not used
from style_transfer.domain.style_transfer_model import BaseStyleTransferModel
from style_transfer.domain.preprocess_data import PreprocessData
from evaluate.evaluator import Text2TextGenerationEvaluator

df = StyleTransferData.data

df_train, df_validation, df_test = BaseData(df).split_train_test(
    test_size=0.15,
    validation_size=0.2,
    random_state=42,
    text_type='joined'
    )

logging.info("Data loaded")
gpt = TransformerStyleTransferModel(
    1,
    model_name="Latino_to_European",
    tokenizer_name='DeepESP/gpt2-spanish',
    batch_size=8,
    cache_dir='cache',
    output_dir='Latino_to_European'
    )
gpt.fit(df_train, df_validation)
logging.info("Model fitted")
