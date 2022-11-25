from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel
from style_transfer.interface.app_transfer_style import TransferStyleApp
import pandas as pd
#TODO : Mettre le tokenizer futur : tokenizer_name='DeepESP/gpt2-spanish',

model = TransformerStyleTransferModel(
    model_name="models/Latino_to_European_GColab",
    tokenizer_name='flax-community/gpt-2-spanish'
    )
app = TransferStyleApp(model)
app.RUN()


