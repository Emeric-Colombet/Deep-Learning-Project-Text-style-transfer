from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel
from IPython.core.display import HTML, display_html
from dataclasses import dataclass
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer
)
import streamlit as st
from bertviz import head_view, model_view


class ExplainabilityApp:

    def __init__(self):

        self.model = AutoModelWithLMHead.from_pretrained("models/Latino_to_European_GColab", output_attentions = True)
        self.tokenizer = AutoTokenizer.from_pretrained('DeepESP/gpt2-spanish')
    

    def RUN(self):
        self._head()
        self._save_model_attention_in_html()
        self._display_model_attention_html()
        self._save_head_attention_in_html()
        self._display_head_attention_html()


    @staticmethod
    def _head():
        st.title('Explainability')
        st.subheader('Showing Attention 🚨')
    
    def _settings_attention(self):

        with open("data/input_text.txt", "r") as file:
            text = file.read()

        inputs = self.tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']
        attention = self.model(input_ids)[-1]
        input_id_list = input_ids[0].tolist() # Batch index 0
        tokens = self.tokenizer.convert_ids_to_tokens(input_id_list)
        return attention, tokens
    
    def _save_head_attention_in_html(self):
        attention, tokens = self._settings_attention()
        html_head_view = head_view(attention, tokens, html_action='return')
        with open("assets/head_view.html", 'w') as file:
            file.write(html_head_view.data)

    @staticmethod
    def _display_head_attention_html():
        with open("assets/head_view.html", "r") as file:
            html_string = file.read()
        st.caption("Head Attention View")
        st.components.v1.html(html_string,height=500, width =300)
    
    def _save_model_attention_in_html(self):
        attention, tokens = self._settings_attention()
        html_model_view = model_view(attention, tokens, html_action='return')
        with open("assets/model_view.html", 'w') as file:
            file.write(html_model_view.data)
    
    @staticmethod
    def _display_model_attention_html():
        with open("assets/model_view.html", "r") as file:
            html_string = file.read()
        st.caption("Model Attention View")
        st.components.v1.html(html_string, height = 1500, width = 1000)


    