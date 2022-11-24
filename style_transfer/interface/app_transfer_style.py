from dataclasses import dataclass
import streamlit as st
import pandas as pd
import numpy as np
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel
from style_transfer.domain.preprocess_data import BaseData
@dataclass
class TransferStyleApp:
    model : TransformerStyleTransferModel

    def RUN(self):
        self._head()
        submit, text_to_submit = self._text_placeholder()
        if submit : 
            encoded_text_to_submit = self._compute_text_preprocessing(text_to_submit)
            predictions,_ = self.model.predict(encoded_text_to_submit)
            self._display_prediction(predictions)

    def transform_style(self,sentence):
        prediction = self.model.predict(sentence)
        return prediction

    def _compute_text_preprocessing(self,text_to_submit):
        pre_process_text_to_submit = BaseData.utils_from_str_to_pandas(text_to_submit)
        encoded_text_to_submit = BaseData.format_df_for_model(pre_process_text_to_submit,text_type="encoded")
        return encoded_text_to_submit

    @staticmethod
    def _configure_page():
        st.set_page_config(
            page_title='Spanish text style transfer',
            page_icon='assets/bandera.jpg')
    
    @staticmethod
    def _head():
        st.title('Spanish text style transfer')
        st.caption('From Latino to European style')
    
    @staticmethod
    def _text_placeholder():
        """ This fonction summurize the first part of our application :
    :text_submission : The area where the user can write all the sentences in Latinamerica style
    
    :submit : The button permitting to generate the prediction by feeding the model with input_sentences
    """
        LAM = "Dios mío. ¿se quedó a dormir?"\
            "Skye, creo que esto está listo. bien."\
            "No tengo idea. no puedo lidiar... he estado en situaciones"
        text_to_submit = st.text_area(
            'Text on which we transfer style', 
            placeholder = LAM,
            height=200,
            help=("Write here your Latinamerica Spanish text, and we will transform it into European Spanish style!  \n" \
                "If you want to separate sentences uses ';' symbol. "))
        submit = st.button('Submit')
        return submit,text_to_submit
    
    @staticmethod
    def _display_prediction(predictions="Output of our model"):
        st.subheader('Prediction :')
        markdow_display = ""
        for sentence in predictions:
            new_line = f"{sentence}  \n"
            markdow_display += new_line
        st.markdown(markdow_display)
