import irc.bot
import irc.strings
import time
import sys
import threading
import random
import json
import os
import subprocess
from pathlib import Path
import re

LABEL_MAP = {
    0: "non-self-intro",
    1: "self-intro",
}
MAX_CHUNK_TOKENS = 510
PERSON_LABEL = "PERSON"

TITLES = re.compile(
    r"\b(Mr|Mrs|Ms|Miss|Dr|Prof|Professor|Senator|Sen|"
    r"Assemblymember|Assemblywoman|Assemblyman|"
    r"Chairman|Chairwoman|Chair|Vice Chair|"
    r"Representative|Rep|Councilmember|Judge|Officer|"
    r"Director|Secretary|Commissioner|Madam|Sir)\.?\s*",
    re.IGNORECASE
)

class ChatBot(irc.bot.SingleServerIRCBot):
    def __init__(self, channel, nickname, server, port=6667):
        super().__init__([(server, port)], nickname, nickname)
        self.channel_name = channel
        self.creator_info = "Pallavi Das and Kasey Liu, CSC 482"
        self.lock = threading.RLock()
        self.timeout_timer = None
        self.model_dir = self.resolve_model_dir()
        self.spacy_model_dir = self.resolve_spacy_model_dir()
        self.classifier = None
        self.spacy_nlp = None
        # Buffer for multi-part classify messages
        self.classify_buffer = {}
        self.classify_timers = {}

    def on_nicknameinuse(self, c, e):
        c.nick(c.get_nickname() + "_")

    def on_welcome(self, c, e):
        c.join(self.channel_name)
        print(f"[{c.get_nickname()}] Connected and joined {self.channel_name}")

    def on_pubmsg(self, c, e):
        msg = e.arguments[0]
        sender = e.source.nick
        
        print(f"[{self.channel_name}] {sender}: {msg}")
        prefix = c.get_nickname() + ":"
        
        if sender in self.classify_buffer and not msg.lower().startswith(prefix.lower()):
            self._append_to_classify_buffer(sender, " " + msg)
            return
        
        if msg.lower().startswith(prefix.lower()):
            text = msg[len(prefix):].strip()
            command = text.lower()
            if self.is_command(command):
                self.do_command(e, text, c)

    def _append_to_classify_buffer(self, sender, text):
        with self.lock:
            self.classify_buffer[sender] = self.classify_buffer.get(sender, "") + text
            if sender in self.classify_timers:
                self.classify_timers[sender].cancel()
            timer = threading.Timer(2.0, self._flush_classify_buffer, args=[sender])
            timer.daemon = True
            timer.start()
            self.classify_timers[sender] = timer

    def _flush_classify_buffer(self, sender):
        with self.lock:
            full_text = self.classify_buffer.pop(sender, "")
            self.classify_timers.pop(sender, None)
        if full_text.strip():
            print(f"[buffer] Processing full classify text from {sender} ({len(full_text)} chars)")
            self.handle_classifier_message(sender, full_text.strip())

    def send_delayed_msg(self, target, msg, delay=2):
        def delayed_send():
            time.sleep(delay)
            self.connection.privmsg(target, msg)
            print(f"[{target}] {self.connection.get_nickname()}: {msg}")
            
        threading.Thread(target=delayed_send, daemon=True).start()

    def is_command(self, cmd):
        return (
            cmd in {"die", "forget", "who are you?", "usage", "users", "classify"}
            or cmd.startswith("classify ")
        )

    def resolve_model_dir(self):
        bot_root = Path(__file__).resolve().parent
        return bot_root / "models" / "bert_self_intro_classifier"

    def resolve_spacy_model_dir(self):
        bot_root = Path(__file__).resolve().parent
        return bot_root / "models" / "spacy_legislative_ner"

    def load_bert_classifier(self, model_dir):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ModuleNotFoundError as ex:
            print(f"BERT classifier unavailable (missing dependency): {ex}")
            return None

        if not model_dir.exists():
            print(f"BERT classifier unavailable (model directory not found): {model_dir}")
            return None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            print(f"Loaded BERT classifier from {model_dir}")
            return {
                "torch": torch,
                "tokenizer": tokenizer,
                "model": model,
                "device": device,
            }
        except Exception as ex:
            print(f"Failed to load BERT classifier from {model_dir}: {ex}")
            return None

    def load_spacy_model(self):
        try:
            import spacy
            print(f"Loading spaCy model from {self.spacy_model_dir}...")
            return spacy.load(self.spacy_model_dir)
        except Exception as ex:
            print(f"Failed to load spaCy model: {ex}")
            return None

    def predict_chunk(self, chunk_text):
        bundle = self.classifier
        torch = bundle["torch"]
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        device = bundle["device"]

        inputs = tokenizer(chunk_text, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

        pred_id = int(torch.argmax(probs).item())
        conf = float(probs[pred_id].item())
        return pred_id, conf

    def chunk_text(self, text):
        tokenizer = self.classifier["tokenizer"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            return [text]

        chunks = []
        for i in range(0, len(token_ids), MAX_CHUNK_TOKENS):
            chunk_ids = token_ids[i : i + MAX_CHUNK_TOKENS]
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
        return chunks

    def classify_text(self, text):
        if self.classifier is None:
            self.classifier = self.load_bert_classifier(self.model_dir)
        if not self.classifier:
            return None, None

        chunks = self.chunk_text(text)
        chunk_preds = [self.predict_chunk(c) for c in chunks]

        if any(pred_id == 1 for pred_id, _ in chunk_preds):
            conf = max(conf for pred_id, conf in chunk_preds if pred_id == 1)
            return LABEL_MAP[1], conf

        conf = max(conf for _, conf in chunk_preds) if chunk_preds else 0.0
        return LABEL_MAP[0], conf

    def normalize_name(self, name):
        name = TITLES.sub("", name).strip()
        name = name.strip(".,;:")
        return name

    def extract_person_names_with_spacy(self, text):
        if self.spacy_nlp is None:
            self.spacy_nlp = self.load_spacy_model()
        if not self.spacy_nlp:
            return set()

        try:
            doc = self.spacy_nlp(text)
            names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        except Exception as ex:
            print(f"spaCy inference failed: {ex}")
            return set()

        processed_names = set()
        for name in names:
            normalized = self.normalize_name(name)
            if normalized and len(normalized) > 1:
                processed_names.add(normalized)
        return processed_names

    def extract_names_and_speaker(self, text):
        names = set()
        if self.spacy_model_dir.exists():
            names = self.extract_person_names_with_spacy(text)
        
        speaker = None
        lower_text = text.lower()
        trigger_phrases = ["my name is ", "my name's ", "i'm ", "i am ", "this is "]
        
        for name in names:
            name_lower = name.lower()
            for trigger in trigger_phrases:
                idx = lower_text.find(trigger)
                if idx != -1:
                    expected_pos = idx + len(trigger)
                    name_pos = lower_text.find(name_lower, expected_pos)
                    # if the name appears within 30 characters after the trigger phrase
                    if name_pos != -1 and (name_pos - expected_pos) < 30:
                        speaker = name
                        break
            if speaker:
                break
                
        # if BERT flagged as self-intro, but triggers failed, guess the first name if available
        if not speaker and names:
            for pat in ["on behalf of", "representing", "here today"]:
                if pat in lower_text:
                    speaker = list(names)[0]
                    break
            
            if not speaker and len(names) == 1:
                speaker = list(names)[0]
        
        return speaker, names

    def handle_classifier_message(self, sender, text):
        label, conf = self.classify_text(text)
        if label is None:
            self.send_to_user(
                sender,
                f"Classifier unavailable. Expected model directory: {self.model_dir}",
                delay=1,
            )
            return
            
        speaker, all_names = self.extract_names_and_speaker(text)
        
        if speaker and speaker in all_names:
            all_names.remove(speaker)
            
        names_str = ", ".join(all_names) if all_names else "none"
        conf_pct = f"{conf * 100:.1f}%"

        if label == "self-intro":
            if speaker:
                response = f"This is a self-introduction (confidence: {conf_pct})! I believe the speaker is {speaker}. Other names mentioned: {names_str}."
            else:
                response = f"This is a self-introduction (confidence: {conf_pct})! However, I could not identify the main speaker. The names involved are: {names_str}."
        else:
            if all_names:
                response = f"I do not think this is a self-introduction (confidence: {conf_pct}). However, I did detect the following names: {names_str}."
            else:
                response = f"I do not think this is a self-introduction (confidence: {conf_pct}), and I did not detect any names."

        self.send_to_user(sender, response, delay=1)

    def cancel_timeout_timer(self):
        if self.timeout_timer and self.timeout_timer.is_alive():
            self.timeout_timer.cancel()
        self.timeout_timer = None

    def send_to_user(self, user, text, delay=2):
        self.send_delayed_msg(self.channel_name, f"{user}: {text}", delay=delay)

    def do_command(self, e, cmd, c):
        sender = e.source.nick
        target = self.channel_name
        normalized = cmd.strip().lower()

        if normalized == "die":
            self.send_delayed_msg(target, f"{sender}: I shall!")
            time.sleep(3) 
            c.quit("Quit: chat-bot")
            sys.exit(0)

        elif normalized == "forget":
            self.send_delayed_msg(target, f"{sender}: forgetting everything")
            with self.lock:
                self.cancel_timeout_timer()
            
        elif normalized in ["who are you?", "usage"]:
            msg1 = f"My name is {c.get_nickname()}. I was created by {self.creator_info}"
            msg2 = 'Use "classify <text>" to run our legislative self-intro and name extraction classifier.'
            msg3 = 'Kasey implemented the BERT self-intro classifier, and Pallavi implemented the finetuned spaCy legislative NER model.'
            self.send_delayed_msg(target, f"{sender}: {msg1}")
            self.send_delayed_msg(target, f"{sender}: {msg2}")
            self.send_delayed_msg(target, f"{sender}: {msg3}")

        elif normalized == "users":
            channel_obj = self.channels.get(self.channel_name)
            if channel_obj:
                users_list = ", ".join(channel_obj.users())
                self.send_delayed_msg(target, f"{sender}: {users_list}")
            else:
                self.send_delayed_msg(target, f"{sender}: I am not in the channel.")

        elif normalized == "classify" or normalized.startswith("classify "):
            text = cmd[len("classify"):].strip() if normalized.startswith("classify") else ""
            if not text:
                self.send_delayed_msg(
                    target,
                    f'{sender}: Usage: "{c.get_nickname()}: classify <text>"',
                )
                return
            
            self._append_to_classify_buffer(sender, text)


if __name__ == "__main__":
    SERVER = "irc.libera.chat"
    PORT = 6667
    CHANNEL = "#CSC482"
    BOTNICK = "dasliu-bot"
    
    print(f"Starting {BOTNICK} on {SERVER}:{PORT} in {CHANNEL}...")
    bot = ChatBot(CHANNEL, BOTNICK, SERVER, PORT)
    try:
        bot.start()
    except KeyboardInterrupt:
        print("\nBot shutting down manually.")
        sys.exit(0)
