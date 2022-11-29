import streamlit as st
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel
from style_transfer.interface.app_transfer_style import TransferStyleApp
st.set_page_config(
    page_title="Running Style Transfer", 
    page_icon="📖",
    layout="wide"
    )
model = TransformerStyleTransferModel(
    model_name="models/Latino_to_European_GColab",
    tokenizer_name='DeepESP/gpt2-spanish'
    )
app_transfer_style = TransferStyleApp(model)
app_transfer_style.RUN()