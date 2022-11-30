"""This module load our style transfer model """

from transformers import (
    AutoModelWithLMHead,
    AutoConfig,
    Trainer,
    AutoTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    AutoModelForCausalLM,
    pipeline
)
import evaluate
from style_transfer.domain.preprocess_data import BaseData
from datasets import Dataset, load_dataset
import logging
from xmlrpc.client import Boolean
import os 
from typing import Tuple
from dataclasses import dataclass 
import tensorflow as tf
import json
import pandas as pd
import numpy as np
from tbparse import SummaryReader

class BaseStyleTransferModel:
    """
    Interface of our StyleTransfer model

    """
    PICKLE_ROOT = "data/models"

    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the model from the training set (X, y)."""
        return "fit"

    
    def predict(self, X: pd.DataFrame):
        """Return predictions. This first definition is for app testing only, 
        be carreful using this function with a real dataframe. """
        return f"This is the prediction {X}"

    @classmethod
    def load(cls) -> Boolean:
        """Load the churn model from the class instance name."""
        pass

    def save(self) -> Boolean:
        """Save the model (pkl format)."""
        pass



class TransformerStyleTransferModel(BaseStyleTransferModel):
    def __init__(self,model_name,tokenizer_name,cache_dir='cache',output_dir='Latino_to_European'):
        """ Constructor of initialized class 
        
        :parameters:
            text_path : Path of our combined dataset. 
            model : Model name who will be retrieved on Huggingface hub.
            batch_size : Classical batch_size.
            cache_dir : Dirname of the directory containing cache.
            output_dir : Dirname of thesaved model
        """
        logging.debug("Loading model & tokenizer")
        self.model = AutoModelWithLMHead.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        
        self.cache_dir = cache_dir
        self.output_dir = output_dir


    def fit(self,df_train : pd.DataFrame, df_eval : pd.DataFrame, epochs: int = 1, batch_size: int = 8) -> 'TransformerStyleTransferModel' :
        trainer = self._build_trainer(df_train,df_eval,epochs,batch_size)
        train_result = trainer.train()
        evaluation = trainer.evaluate()
        print(f"Evaluation : {evaluation}")
        logging.debug("Trained")
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)
        trainer.save_metrics("eval",evaluation)
        trainer.save_model()
        return trainer

    def predict(self, df_to_predict: pd.DataFrame, compute_metric: Boolean = False) -> Tuple[list,dict] :
        """     Make a prediction on the incoming dataframe. This one must be have at least a column named encoded_latinamerica.
        If the compute_metric argument is set to True, this function will try to compute the bleu metric. The column "text_spain" must exist.
        """ 
        tokenized_df_to_predict_encoded = self.tokenizer(df_to_predict["encoded_latinamerica"].to_list(),padding=True, return_tensors="pt")
        encoded_predictions = self.model.generate(input_ids=tokenized_df_to_predict_encoded.input_ids,attention_mask=tokenized_df_to_predict_encoded.attention_mask,max_new_tokens=512)
        input_size = len(tokenized_df_to_predict_encoded.input_ids)
        output_size = len(encoded_predictions)
        if input_size != output_size:
            raise Exception(f"Input size must be same as the output size. Here input size is {input_size}, and output size is {output_size}.")
        encoded_predictions_detokenized = self.tokenizer.batch_decode(encoded_predictions,skip_special_tokens=True)
        clean_predictions = BaseData.utils_decode_model_output(encoded_predictions_detokenized)
        if compute_metric :
            dict_results = self._calculate_bleu_score(references=df_to_predict["text_spain"].to_list(),predictions=clean_predictions)
        else : 
            dict_results = None
        return clean_predictions,dict_results
        
    def _calculate_bleu_score(self,references : list, predictions : list) -> dict: 
        """ Take references list and prediction list, and compute the bleu metric"""
        google_bleu = evaluate.load("google_bleu")
        bleu_score = google_bleu.compute(predictions=predictions, references=references)
        return bleu_score

    def _build_trainer(self, df_train: pd.DataFrame, df_eval: pd.DataFrame, epochs : int, batch_size: int) -> Trainer:
        """Private method in order to build a trainer object. 
        """
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        logging.debug("Loading text dataset")
        ds_train = Dataset.from_pandas(df_train,preserve_index=False) 
        tokenized_training_dataset = self._map_tokenizer_on_dataset(ds_train,'encoded_latinamerica_spain')
        ds_eval = Dataset.from_pandas(df_eval,preserve_index=False) 
        tokenized_eval_dataset = self._map_tokenizer_on_dataset(ds_eval,'encoded_latinamerica_spain')
        logging.debug("Loading Training arguments")
        training_args = TrainingArguments(
            output_dir = self.output_dir,
            num_train_epochs = epochs,
            per_device_train_batch_size = batch_size,
            warmup_steps = 500,
            save_steps = 2000,
            logging_steps = 10
        )
        trainer = Trainer(
            model = self.model,
            args = training_args,
            data_collator = data_collator,
            train_dataset = tokenized_training_dataset,
            eval_dataset = tokenized_eval_dataset
        )
        logging.info("Everything loaded well")
        return trainer


    def _map_tokenizer_on_dataset(self, dataset: Dataset, column_name: str) -> Dataset:

        """This function will implement a mapping on the Dataset, using a lambda function to select 
        on which column it will compute the tokenization
        
        :parameters :
            dataset : The dataset, it can be ds_train, ds_validation, ds_test
            column_name : The column on which we want to compute tokenization. (Ex : 'encoded_latinamerica_spain', or 'encoded_latinamerica') 
        
        :returns : 
            encoded_dataset : The dataset encoded. 

        """
        encoded_dataset = dataset.map(
            lambda examples : self.tokenizer(examples[f"{column_name}"],padding="max_length", max_length=512),
            batched=True)
        return encoded_dataset

    def _transpose (self,input_sentence,generator):
        raw_prediction = generator('<s>'+input_sentence + '</s>>>>><p>')
        clean_prediction = raw_prediction[0]['generated_text'].split('</s>>>>><p>')[1].split('</p>')[0]
        return clean_prediction
    
    #TODO : Faire de cette méthode une méthode normale qui utilise self.model_name/logs/ Bleu ou autre pour retrouver les datas.
    @staticmethod
    def retrieve_model_logs(path : str = "jupyter notebook/logs/Nov26_23-23-44_b8eb83221afe", method : str = "train/loss") -> pd.DataFrame :
        """ This method accept a list of actions, and if ok retrieve model's logs  from given folder path
        :parameters :
            method : Can be `Bleu` or `train/loss`. `Bleu` retrieve model's Bleu score on Test dataset. 
                        `train/loss` retrieve model's train/loss score from training and evaluating part. 
        :returns : 
            Dataframe containing information
        """

        acceptable_actions = ['Bleu','train/loss']
        if method not in acceptable_actions : 
            raise ValueError(f"Please provide an acceptable method. Not {str(method)}")
        if method == "Bleu" :
            logging.debug("Retrieving model's logs Bleu")
            google_bleu_json_path = "models/Latino_to_European_GColab/logs/bleu_score_test_dataset.json"
            with open(google_bleu_json_path) as f:
                google_bleu_score = json.load(f)
            return round(float(google_bleu_score["google_bleu"]),3)
            
        if method == "train/loss":
            logging.debug("Retrieving model's logs train/loss")
            reader = SummaryReader(path) 
            all_log_dataframe = reader.scalars
            train_by_loss = all_log_dataframe[all_log_dataframe["tag"].str.contains("train/loss")]
            return train_by_loss


class Seq2seqStyleTransferModel:

    def __init__(self, vocab_size = 30000, max_length = 150,
                 embed_size = 300, output_dir = "models/my_model"):
        
        """ Constructor of initialized class 
        
        :parameters:
            vocab_size : Number of words/tokens in our dictionary. 
            max_length : Maximal length of sentences fed in the model.
            embed_size : Embedding matrix size.
            output_dir : Dirname of the saved model
        """

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_size = embed_size
        self.output_dir = output_dir


    def model_architecture(self, df_cleaned : pd.DataFrame, df_train, df_validation):
        """ Function establishing the model architecture of our seq2seq with attention

        :parameter:
            df_cleaned : Preprocessed DataFrame containing equivalent sentences
                         between latin american spanish and european spanish
            df_train : Train Split of DataFrame
            df_validation : Validation Split of DataFrame
        :return:
            model : model containing the model architecture
            X_train: Tensor object of the latin american sentences 
                     for training to input in encoder layer
            X_train_dec: Tensor object of the euro spanish sentences
                         for training to input in decoder layer
            X_valid: Tensor object of the latin american sentences 
                     for validation to input in encoder layer
            X_valid_dec: Tensor object of the euro spanish sentences
                         for validation to input in decoder layer
            y_train: Target european spanish sentences for training
            y_valid: Target european spanish sentences for validation
        
        """

        #TextVectorization layers that apply a lowercase standardization
        #And Text Tokenization
        text_vec_layer_lat = tf.keras.layers.TextVectorization(self.vocab_size,\
                                         output_sequence_length = self.max_length,\
                                         standardize = "lower")
        text_vec_layer_es = tf.keras.layers.TextVectorization(self.vocab_size,\
                                         output_sequence_length = self.max_length,\
                                             standardize = "lower")
        text_vec_layer_lat.adapt(df_cleaned["text_latinamerica"])
        text_vec_layer_es.adapt([f"startofseq {s} endofseq" for s in df_cleaned["text_spain"]])

        #Creating Tensor constant objects from our split dataset
        X_train = tf.constant(df_train["text_latinamerica"])
        X_valid = tf.constant(df_validation["text_latinamerica"])
        X_train_dec = tf.constant([f"startofseq {s}" for s in df_train["text_spain"]])
        X_valid_dec = tf.constant([f"startofseq {s}" for s in df_validation["text_spain"]])
        
        y_train = text_vec_layer_es([f"{s} endofseq" for s in df_train["text_spain"]])
        y_valid = text_vec_layer_es([f"{s} endofseq" for s in df_validation["text_spain"]])


        #Encoder - Decoder Embeddings layers
        encoder_inputs = tf.keras.layers.Input(shape = [], dtype = tf.string)
        decoder_inputs = tf.keras.layers.Input(shape = [], dtype = tf.string)

        encoder_input_ids = text_vec_layer_lat(encoder_inputs)
        decoder_input_ids = text_vec_layer_es(decoder_inputs)
        encoder_embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embed_size,
                                                    mask_zero=True, 
                                                    embeddings_initializer = "glorot_uniform")
        decoder_embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embed_size,
                                                    mask_zero=True, 
                                                    embeddings_initializer = "glorot_uniform")
        encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
        decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

        #Encoder : Bidirectional LSTM Layer
        tf.random.set_seed(42)  
        encoder = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(512, return_sequences=True, return_state=True, dropout = 0.1))

        encoder_outputs, *encoder_state = encoder(encoder_embeddings)
        encoder_state = [tf.concat(encoder_state[::2], axis=-1),  
                 tf.concat(encoder_state[1::2], axis=-1)]  

        #Decoder : LSTM Layer
        decoder = tf.keras.layers.LSTM(1024, return_sequences=True, dropout =0.1)
        decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)
        
        #Attention Layer applied on the decoder-encoder outputs
        attention_layer = tf.keras.layers.Attention()
        attention_outputs = attention_layer([decoder_outputs, encoder_outputs])

        #Output Layer is a Dense one
        output_layer = tf.keras.layers.Dense(self.vocab_size, activation="softmax")
        Y_proba = output_layer(attention_outputs)

        model = tf.keras.Model(inputs=[encoder_inputs, decoder_inputs],
                       outputs=[Y_proba])

        return model, X_train, X_train_dec, X_valid, X_valid_dec, y_train, y_valid

    def fit(self, model, X_train, X_train_dec, X_valid, X_valid_dec, y_train, y_valid):
        """ Fit method
        :parameters:
            model: model containing the model architecture
            X_train: Tensor object of the latin american sentences 
                     for training to input in encoder layer
            X_train_dec: Tensor object of the euro spanish sentences
                         for training to input in decoder layer
            X_valid: Tensor object of the latin american sentences 
                     for validation to input in encoder layer
            X_valid_dec: Tensor object of the euro spanish sentences
                         for validation to input in decoder layer
            y_train: Target european spanish sentences for training
            y_valid: Target european spanish sentences for validation
        :return:
            model: fitted model

        """
        
        opt = tf.keras.optimizers.Nadam(learning_rate=0.001)
        model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
                     metrics=["accuracy"])
        model.fit((X_train, X_train_dec), y_train, epochs=10,
                    validation_data=((X_valid, X_valid_dec), y_valid))
            
        model.save(self.output_dir)

        return model

    def _load_model(self):
        """Hidden method to load the saved model from output directory
        """
        model = tf.keras.models.load_model(self.output_dir)
        model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
                      metrics=["accuracy"])
       
        return model


    def translate(self,sentence_lat):
        """Translate method to decode the output given by the model
        
        :parameters:
            sentence_lat: sentence in latin american to feed in the model
                          to convert to an european spanish sentence
        :return:
            translation: sentence in european spanish deducted by the encoder-decoder model
        
        """

        model = self._load_model()

        translation = ""
        for word_idx in range(self.max_length):
            X = np.array([sentence_lat]) #encoder input
            X_dec = np.array(["startofseq " + translation]) #decoder input
            y_proba = model.predict((X, X_dec))[0, word_idx] #last token's probas
            predicted_word_id = np.argmax(y_proba)
            #4th index is corresponding to the text_vec_layer_es Layer
            predicted_word = model.layers[4].get_vocabulary()[predicted_word_id]
            if predicted_word == "endofseq":
                break
        translation += " " + predicted_word
        return translation
    