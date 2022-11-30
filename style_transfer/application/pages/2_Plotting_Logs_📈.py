import streamlit as st
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel
from style_transfer.interface.app_monitor import AppModelMonitor
st.set_page_config(
    page_title="Plotting Logs", 
    page_icon="📈",
    layout="wide"
    )
model = TransformerStyleTransferModel(
    model_name="models/Latino_to_European",
    tokenizer_name='DeepESP/gpt2-spanish'
    )
app_monitor = AppModelMonitor(model)
app_monitor.RUN()