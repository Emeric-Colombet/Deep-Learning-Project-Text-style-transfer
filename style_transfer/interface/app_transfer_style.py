from dataclasses import dataclass
import streamlit as st
from gtts import gTTS
import base64
import os
import pandas as pd
import numpy as np
from style_transfer.domain.style_transfer_model import TransformerStyleTransferModel
from style_transfer.domain.preprocess_data import BaseData




@dataclass
class TransferStyleApp:
    model: TransformerStyleTransferModel

    def RUN(self):
        self._head()
        submit, text_to_submit = self._text_placeholder()
        if submit:
            encoded_text_to_submit = self._compute_text_preprocessing(text_to_submit)
            predictions,_ = self.model.predict(encoded_text_to_submit)
            self._display_prediction(predictions)
            self._play_text_to_speech(predictions, region='spain', auto_play=True)

    def transform_style(self, sentence):
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
        st.subheader('From Latino to European style 💃')
    
    @staticmethod
    def _text_placeholder():
        """ This fonction summurize the first part of our application :
        :text_submission : The area where the user can write all the sentences in Latinamerica style

        :submit : The button permitting to generate the prediction by feeding the model with input_sentences
        """
        placeholder_latinamerica = '''Dios mío. ¿se quedó a dormir?
            Skye, creo que esto está listo. bien
            No tengo idea. no puedo lidiar... he estado en situaciones
        '''
        placeholder_spain = '''Madre mía, ¿se quedó a dormir?
            Skye, creo que esto está listo. vale.
            Yo qué coño sé. no puedo... a veces me ha pasado
        '''
        text_to_submit = st.text_area(
            "Write here your Latino Spanish text, and we will transform it into European Spanish style! ¡Venga!",
            value=placeholder_latinamerica,
            height=200,
            help="If you want to separate sentences uses ';' symbol."
        )
        submit = st.button('Submit')
        return submit, text_to_submit

       
    @staticmethod
    def _display_prediction(predictions="Output of our model"):
        st.subheader('Prediction :')
        markdow_display = ""
        for sentence in predictions:
            new_line = f"{sentence}  \n"
            markdow_display += new_line
        st.markdown(markdow_display)

    @classmethod
    def _play_text_to_speech(cls, text: str, region='spain', auto_play=True):
        """Reads text provided with a specific Spanish accent

        :parameters:
            text: Text to be read
            region: Accent to be applied on speech: 'spain' for European Spanish, else Latinamerican Spanish
        """

        LANG = 'es'
        TTS_FILE = 'text_to_speech_tmp.mp3'

        if region == 'spain':
            tld = 'es'
        else:
            tld = 'com'

        tts = gTTS(text, lang=LANG, tld=tld)
        tts.save(TTS_FILE)

        if auto_play:
            cls._play_audio_auto(TTS_FILE)
        else:
            audio_file = open(TTS_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio / ogg')

        os.remove(TTS_FILE)

    @classmethod
    def _play_audio_auto(cls, file_path: str):
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(
                md,
                unsafe_allow_html=True,
            )

        

