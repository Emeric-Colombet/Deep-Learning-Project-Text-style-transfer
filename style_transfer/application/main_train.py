import logging
import argparse
from style_transfer.infrastructure.style_transfer_data import StyleTransferData
from style_transfer.domain.preprocess_data import BaseData
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel

logging.getLogger().setLevel(logging.INFO)
PARSER = argparse.ArgumentParser(
    description='Compute training from a csv file')
PARSER.add_argument("-d","--debug", action='store_true',
                    help='Activate debug logs')
PARSER.add_argument("-b", "--BleuScore", action='store_true',
                    help="After training, this option compute automatically the bleu score on the df_test dataset.")
PARSER.add_argument("-e", "--epochs", type=str,
                    help="Choose the number of epochs to train the model. 1 Epoch = 2h, default is 1",
                    default=1)
args = PARSER.parse_args()

if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)


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
    output_dir='models/Latino_to_European'
    )
gpt.fit(df_train, df_validation,epochs=int(args.epochs),batch_size=8)
logging.info("Model fitted")

if args.BleuScore :
    print(args.BleuScore)
    quit()
