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
from datasets import Dataset, load_dataset
import logging
from xmlrpc.client import Boolean
import os 
from dataclasses import dataclass 
import pickle
import pandas as pd

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



#TODO : Enlever text_path et donner à la place df_train ainsi que df_eval


class TransformerStyleTransferModel(BaseStyleTransferModel):
    def __init__(self,epochs,model_name,tokenizer_name,batch_size=8,cache_dir='cache',output_dir='Latino_to_European'):
        """ Constructor of initialized class 
        
        :parameters:
            text_path : Path of our combined dataset. 
            model : Model name who will be retrieved on Huggingface hub.
            batch_size : Classical batch_size.
            cache_dir : Dirname of the directory containing cache.
            output_dir : Dirname of thesaved model
        """
        self.epochs = epochs
        self.model = AutoModelWithLMHead.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.output_dir = output_dir


    def fit(self,df_train : pd.DataFrame, df_eval : pd.DataFrame) -> 'TransformerStyleTransferModel' :
        trainer = self._build_trainer(df_train,df_eval)
        train_result = trainer.train()
        logging.debug("Trained")
        #TODO : Log eval metrics too
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model()
        return trainer

    def predict(self,df_test : pd.DataFrame) -> pd.DataFrame : 
        generator = pipeline('text-generation',model=self.model, tokenizer=self.tokenizer)
        return None 
    def _build_trainer(self,df_train : pd.DataFrame, df_eval : pd.DataFrame) -> 'TransformerStyleTransferModel':
        """Private method in order to build a trainer object. 
        """
        logging.debug("Loading model & tokenizer")
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        logging.debug("Loading text dataset")
        ds_train = Dataset.from_pandas(df_train,preserve_index=False) 
        tokenized_training_dataset = ds_train.map(self._tokenize_function, batched=True)
        ds_eval = Dataset.from_pandas(df_eval,preserve_index=False) 
        tokenized_eval_dataset = ds_eval.map(self._tokenize_function, batched=True)
        logging.debug("Loading Training arguments")
        training_args = TrainingArguments(
            output_dir = self.output_dir,
            num_train_epochs = self.epochs,
            per_device_train_batch_size = self.batch_size,
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
    def _tokenize_function(self,examples):
        return self.tokenizer(examples["combined"], padding="max_length", max_length=512)

    def _transpose (input_sentence,generator):
        raw_prediction = generator('<s>'+input_sentence + '</s>>>>><p>')
        clean_prediction = raw_prediction[0]['generated_text'].split('</s>>>>><p>')[1].split('</p>')[0]
        return clean_prediction