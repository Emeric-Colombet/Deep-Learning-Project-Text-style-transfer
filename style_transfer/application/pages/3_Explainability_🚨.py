from style_transfer.interface.app_explainability import ExplainabilityApp
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel

import streamlit as st

st.set_page_config(
    page_title="Explainability", 
    page_icon="🚨",
    layout="wide"
    )


app_explainability = ExplainabilityApp()
app_explainability.RUN()