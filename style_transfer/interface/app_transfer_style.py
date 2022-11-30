from dataclasses import dataclass
import streamlit as st
from gtts import gTTS
import base64
import os

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
            predictions, _ = self.model.predict(encoded_text_to_submit)
            self._display_prediction(predictions)
            speech_format_prediction = self._from_list_of_words_to_string(predictions)
            self._play_text_to_speech(speech_format_prediction, region='spain', auto_play=True)

    def transform_style(self, sentence):
        prediction = self.model.predict(sentence)
        return prediction

    def _compute_text_preprocessing(self, text_to_submit):
        pre_process_text_to_submit = BaseData.utils_from_str_to_pandas(text_to_submit)
        encoded_text_to_submit = BaseData.format_df_for_model(pre_process_text_to_submit, text_type="encoded")
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
        """This function summarizes the first part of our application

        :returns:
            submit: The button permitting to generate the prediction by feeding the model with input_sentences
            text_to_submit: Latinamerica test to transfer style
        """
        option = st.selectbox(
            'Choose a phrase in Latin-American style to get started and we will transform it into European Spanish style!',
            (
            'El valor no significa una mierda a menos que tengas el valor de enfrentar al que se porte como un imbécil.',
            'Y buenas noches, huéspedes. Deben estar muy mal y deben extrañar mucho a sus padres.',
            'Espera, no. ¿Qué pasó? Entré a la habitación equivocada.',
            'Buen trabajo. Estuvo bien. Eso fue una locura. ¿No lo fue?',
            'Es estupendo, apuesto, y todos lo aman. Cuando se va a su casa,  te enteras de que está casado.',
            'Se ve estupendo. Lo están entendiendo. Genial. Vamos a las caídas de espalda.',
            '¡Jódete! ¡Jódete! ¡No me importa un carajo lo que estoy tocando!',
            '"Bash", por favor. Para que sepan, los cheques vienen de él, así que sean amables.',
            '¿Muchos nombres? ¿Con quién estuve compartiendo mi cama?',
            'Linda. Te vio hacer eso.',
            'Es buena en su trabajo. El punto es que Antoine era un cliente joven y atractivo,',
            'Qué rico. ¿Dónde tenías escondido a este chico tan lindo?',
            'Chicas, vengan. ¡Vamos! ¡Se ve estupenda!',
            'Genial. Es bonita. Digo...',
            'Está bien. Me preocupa que ustedes dos vivan solas aquí.',
            '¡Dios, estás embarazada! Cálmate.',
            'Está bien. Quizás tengan razón.',
            '¿Pensaron que podían agarrarme? ¡Haría falta un ejército!',
            'Me acaban de robar, así que quiero las mejores cámaras que tengan.',
            '¿Ya están bebiendo? ¡Sí!',
            'Pero qué descaro el de ese hombre. Bill nunca tuvo sentido común.',
            '¡Trabaron la puerta! ¡Mierda! Vamos. Es solo un auto. ¡Agárrense de algo!',
            'Si necesitan saber el nombre de alguien, solo preguntenme.',
            '¡Cielos! Bueno, tontos. Ahora pueden encender sus teléfonos,',
            '¿Se hicieron pasar por Tareq? Le arruinaron la vida a Ruqayya, ¿entienden?',
            'Es un imbécil, y estoy harto de toda esta mierda vegana.',
            'Genial. Perdón, ¿ese es el tipo del que hablas?'
            ))
        
        text_to_submit = st.text_area(
            "...or you can also write your own below, ¡venga!:",
            value=option,
            height=200,
            help="If you are including separate statements, please use the ';' symbol to separate them."
        )
        submit = st.button('Submit')
        return submit, text_to_submit

    @classmethod
    def _display_prediction(cls, predictions: str = "Output of our model"):
        st.subheader('Prediction :')
        markdown_display = cls._from_list_of_words_to_string(predictions)
        st.markdown(markdown_display)

    @staticmethod
    def _from_list_of_words_to_string(list_of_words: list) -> str:
        markdown_display = ""
        for sentence in list_of_words:
            new_line = f"{sentence}  \n"
            markdown_display += new_line
        return markdown_display

    @classmethod
    def _play_text_to_speech(cls, text: str, region='spain', auto_play=True):
        """Reads text provided with a specific Spanish accent

        :parameters:
            text: Text to be read
            region: Accent to be applied on speech: 'spain' for European Spanish, else Latinamerican Spanish
        """

        LANG = 'es'
        TTS_FILE = 'data/text_to_speech_tmp.mp3'

        if region == 'spain':
            tld = 'es'
        else:
            tld = 'com'

        tts = gTTS(text, lang=LANG, tld=tld)
        tts.save(TTS_FILE)

        if auto_play:
            cls._autoplay_audio(TTS_FILE)
        else:
            audio_file = open(TTS_FILE, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format='audio / ogg')

        os.remove(TTS_FILE)

    @classmethod
    def _autoplay_audio(cls, file_path: str):
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
