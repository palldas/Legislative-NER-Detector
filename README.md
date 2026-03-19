# Legislative NER Detector

- Pallavi Das, Kasey Liu
- CSC 482 Final Project


Before running the code, make sure to download the DH2024_Corpus_Release folder from Hugging Face: https://huggingface.co/datasets/iatpp/digitaldemocracy-2015-2018/blob/main/DH2024_Corpus_Release.zip

To train the trained BERT classifier bot (classifies as self intro or not), run python3 train_bert.py

To train the spaCy legislative name detector used by `bot.py`, run python3 train_spacy_ner.py
