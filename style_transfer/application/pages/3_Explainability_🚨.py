from style_transfer.interface.app_explainability import ExplainabilityApp
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel

import streamlit as st

st.set_page_config(
    page_title="Explainability", 
    page_icon="🚨",
    layout="wide"
    )

#model = TransformerStyleTransferModel(
#    model_name="models/Latino_to_European_GColab",
#    tokenizer_name='DeepESP/gpt2-spanish'
#    )

#model = model.model
#tokenizer = model.tokenizer

app_explainability = ExplainabilityApp()
app_explainability.RUN()