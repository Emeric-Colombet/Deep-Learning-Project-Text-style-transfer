# Text Style Transfer Salima Stephanie Emeric

This project aims to transform a Latina Spanish text in the style of European Spanish. 

# Download Style Transfer Model 
If you want to get the files of the trained model, please send a mail to emeric.colombet@2020.icam.fr and put it in `models/Latino_to_European`
## Installation

Clone the project and create a virtual environment with all the dependencies.

	git clone git@gitlab.com:yotta-academy/mle-bootcamp/projects/dl-projects/project-2-fall-2022/text-style-transfer-salima-stephanie-emeric.git
    cd text-style-transfer-salima-stephanie-emeric/
    python3 -m venv .style_transfer_env
    source .style_transfer_env/bin/activate
    pip install -r requirements.txt
    . activate.sh
## Run Training:
### Run GPT2 training :
(The following command run a training for 1 epoch, it  take 2h per epochs)    
    python3 style_transfer/application/main_train.py --epochs 1 -b 
### Run Seq2Seq NN training : 
    python3 style_transfer/application/main_inference_seq2seq.py
## Run Application: 
    
    streamlit run style_transfer/application/Home_\ 👋.py

## N.B 
    You can find in the models/logs :
        - Output_model.csv : Predictions for the 500 first rows of the GPT2 Transformers
        - bleu_score_test_dataset.json : The output of the bleu score calculation for those 500 first rows
        - Latest_run.csv : The training loss logs of our model. (It was really difficult to calculate bleu score over the training steps with the Trainer object from Hugging Face.)