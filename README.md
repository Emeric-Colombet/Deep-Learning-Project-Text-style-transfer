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
    
    python3 style_transfer/application/main_train.py
## Run Application: 
    
    streamlit run style_transfer/application/Home_\ 👋.py
