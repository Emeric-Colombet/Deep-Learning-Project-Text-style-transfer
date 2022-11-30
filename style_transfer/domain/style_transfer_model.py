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
