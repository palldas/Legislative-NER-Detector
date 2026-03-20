# Legislative NER Detector

- Pallavi Das, Kasey Liu
- CSC 482 Final Project


Before running the code, make sure to download the DH2024_Corpus_Release folder from Hugging Face: https://huggingface.co/datasets/iatpp/digitaldemocracy-2015-2018/blob/main/DH2024_Corpus_Release.zip

To run our IRC bot interface, run `python3 bot.py`.

Our bot.py uses the trained models in this directory, but to retrain, run the following:
- To train the trained BERT classifier bot (classifies as self intro or not), run `python3 train_bert.py`
- To train the spaCy legislative name detector used by bot.py, run `python3 train_spacy_ner.py`

To run the experiments which we got the metrics like accuracy, precision, recall, and F1 from:
- To run the BERT classifier experiment, run `python3 bert_trained_vs_untrained_experiment.py`
- To run the spaCy NER experiment, run `python3 spacy_trained_vs_untrained_experiment.py`