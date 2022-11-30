import streamlit as st
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel
from style_transfer.interface.app_transfer_style import TransferStyleApp
from style_transfer.interface.app_monitor import AppModelMonitor
st.set_page_config(layout="wide")
model = TransformerStyleTransferModel(
    model_name="models/Latino_to_European_GColab",
    tokenizer_name='DeepESP/gpt2-spanish'
    )
app_transfer_style = TransferStyleApp(model)
app_monitor = AppModelMonitor(model)


PAGES = {
    "Style transfer": app_transfer_style,
    "monitor my model": app_monitor
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.RUN()
