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
    text_type='combined'
    )

logging.info("Data loaded")
gpt = TransformerStyleTransferModel(
    model_name="DeepESP/gpt2-spanish",
    tokenizer_name='DeepESP/gpt2-spanish',
    cache_dir='cache',
    output_dir='models/Latino_to_European_GColab'
    )
gpt.fit(df_train, df_validation,epochs=1,batch_size=8)
logging.info("Model fitted")
