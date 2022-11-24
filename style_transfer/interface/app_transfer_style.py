from dataclasses import dataclass
import streamlit as st

from style_transfer.domain.style_transfer_model import BaseStyleTransferModel

from gtts import gTTS
import base64
import os


@dataclass
class TransferStyleApp:
    model: BaseStyleTransferModel

    def RUN(self):
        self._head()
        submit, text_submission = self._text_placeholder()
        if submit:
            prediction = self.model.predict(text_submission)
            self._display_prediction(prediction)
            self._play_text_to_speech(prediction, region='spain', auto_play=True)

    def transform_style(self, sentence):
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
        st.subheader('From Latino to European style 💃')
    
    @staticmethod
    def _text_placeholder():
        placeholder_latinamerica = '''Dios mío. ¿se quedó a dormir?
Skye, creo que esto está listo. bien
No tengo idea. no puedo lidiar... he estado en situaciones
        '''
        placeholder_spain = '''Madre mía, ¿se quedó a dormir?
Skye, creo que esto está listo. vale.
Yo qué coño sé. no puedo... a veces me ha pasado
        '''
        text_submission = st.text_area(
            "Write here your Latino Spanish text, and we will transform it into European Spanish style! ¡Venga!",
            value=placeholder_latinamerica,
            height=150,
            help="Write here Spanish text with the style of Latin America"
        )
        submit = st.button('Submit')
        return submit, text_submission
    
    @staticmethod
    def _display_prediction(prediction="Output of our model"):
        st.subheader('Prediction :')
        st.markdown(f"**{prediction}**")

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
