from dataclasses import dataclass
import streamlit as st
import pandas as pd
import numpy as np
from style_transfer.domain.style_transfer_model import BaseStyleTransferModel

@dataclass
class TransferStyleApp:
    model : BaseStyleTransferModel
    def RUN(self):
        self._head()
        submit, text_submission = self._text_placeholder()
        self._display_prediction()
        if submit : 
            prediction = self.model.predict(text_submission)
            self._display_prediction(prediction)


    def transform_style(self,sentence):
        prediction = self.model.predict(sentence)
        return prediction
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
        LAM = '''    Dios mío. ¿se quedó a dormir?
    Skye, creo que esto está listo. bien.
    No tengo idea. no puedo lidiar... he estado en situaciones
    '''
        ES = '''    Madre mía, ¿se quedó a dormir?
        Skye, creo que esto está listo. vale. 
        Yo qué coño sé. no puedo... a veces me ha pasado
        '''
        text_submission = st.text_area('Text on wich we transfer style', placeholder = LAM,height=150,help="Write here your Latino Spanish text, and we will transform it into European Spanish style!!")
        submit = st.button('Submit')
        return submit,text_submission
    @staticmethod
    def _display_prediction(prediction="Output of our model"):
        st.subheader('Prediction :')
        st.markdown(f"**{prediction}**")
