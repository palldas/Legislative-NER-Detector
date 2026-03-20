# Legislative NER Detector

- Pallavi Das, Kasey Liu
- CSC 482 Final Project

This file describes the environment and setup required to run the Legislative NER Detector project.

## Create and Activate a Virtual Environment:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

## Install Dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
python3 -m spacy download en_core_web_trf
```

## Usage
After installing dependencies, navigate to the `code` directory and run the bot interface:
```bash
cd code
python3 bot.py
```
Address the bot with `dasliu-bot: classify <text>` in the channel.

> [!NOTE]
> **Performance Note**: The very first `classify` command you send after starting the bot will be slower (taking ~30-60 seconds). This is because the bot "lazy-loads" the heavy BERT and spaCy transformer models into memory only when they are first needed. Once loaded, all subsequent requests will be near-instant.

## Re-Training the Models (Optional)
Our bot.py uses the trained models in this directory, but to retrain, run the following:
- Before running the code, make sure to download the DH2024_Corpus_Release folder from Hugging Face: https://huggingface.co/datasets/iatpp/digitaldemocracy-2015-2018/blob/main/DH2024_Corpus_Release.zip
- To train the trained BERT classifier bot (classifies as self intro or not), run `python3 train_bert.py`
- To train the spaCy legislative name detector used by bot.py, run `python3 train_spacy_ner.py`

## Experiments
To run the experiments (Accuracy, Precision, Recall, and F1), ensure you are in the `code` directory:
```bash
cd code
python3 bert_trained_vs_untrained_experiment.py
python3 spacy_trained_vs_untrained_experiment.py
```

## Project Structure
- `bot.py`: Main IRC chatbot implementation.
- `train_spacy_ner.py`: Fine-tuning script for the spaCy transformer model.
- `train_bert.py`: Fine-tuning script for the BERT classifier.
- `requirements.txt`: List of all Python dependencies.
- `models/`: Contains the saved fine-tuned model weights (BERT and spaCy).
- `results/`: Contains the metrics and evaluation outputs.
- `code/data/`: Contains the labeled legislative dataset (CSV).