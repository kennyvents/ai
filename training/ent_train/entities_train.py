import json
import os
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset

# === Параметры ===
MODEL_NAME = 'DeepPavlov/rubert-base-cased'
LABELS = ['O', 'B-TYPE', 'I-TYPE',
          'B-FSIZE','I-FSIZE','B-SSIZE', 'I-SSIZE',
          'B-MATERIAL', 'I-MATERIAL', 'B-THICKNESS', 'I-THICKNESS', 'B-FLANGE',  'B-LENGTH',
           'B-SECONDL', 'I-SECONDL', 'B-TYPENUM',
          'Ø', 'φ', 'D', 'd', '⌀']
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
MAX_LEN = 128

# === Загрузка и подготовка данных ===
with open("voz_pos.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def encode(example):
    tokens = example["tokens"]
    labels = example["ner_tags"]
    enc = tokenizer(tokens, is_split_into_words=True,
                    truncation=True, max_length=MAX_LEN, padding="max_length")
    word_ids = enc.word_ids()
    label_ids, prev = [], None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != prev:
            label_ids.append(LABEL2ID[labels[word_idx]])  # первый сабворд слова
        else:
            label_ids.append(-100)  # последующие сабворды игнорим в лоссе
        prev = word_idx
    enc["labels"] = label_ids
    return enc

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

flang_pairs = [
    'т2.т2', 'т2.т3', 'т2.ш2', 'т2.ш3', 'т2.н', 'т2.крф', 'т2.пф', 'т2.г',
    'т3.т2', 'т3.т3', 'т3.ш2', 'т3.ш3', 'т3.н', 'т3.крф', 'т3.пф', 'т3.г',
    'ш2.т2', 'ш2.т3', 'ш2.ш2', 'ш2.ш3', 'ш2.н', 'ш2.крф', 'ш2.пф', 'ш2.г',
    'ш3.т2', 'ш3.т3', 'ш3.ш2', 'ш3.ш3', 'ш3.н', 'ш3.крф', 'ш3.пф', 'ш3.г',
    'н.т2',  'н.т3',  'н.ш2',  'н.ш3',  'н.н',  'н.крф',  'н.пф',  'н.г',
    'крф.т2', 'крф.т3', 'крф.ш2', 'крф.ш3', 'крф.н', 'крф.крф', 'крф.пф', 'крф.г',
    'пф.т2', 'пф.т3', 'пф.ш2', 'пф.ш3', 'пф.н', 'пф.крф', 'пф.пф', 'пф.г',
    'г.т2',  'г.т3',  'г.ш2',  'г.ш3',  'г.н',  'г.крф',  'г.пф',  'г.г'
]

diameters = ["Ø","ø","O","o","D","d","⌀","φ","Ф","ф","∅"]

special_tokens = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0',
                  'воздуховод',
                  'AISI 304', 'AISI 316', 'AISI 430', 'ч.мет',
                  *flang_pairs, *diameters
                  ]
tokenizer.add_tokens(special_tokens)



dataset = Dataset.from_list(data)
encoded_dataset = dataset.map(encode)

# === Загрузка модели ===
model = BertForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

model.resize_token_embeddings(len(tokenizer))

# === Аргументы обучения ===
training_args = TrainingArguments(
    # output_dir="../voz_pos/",
    output_dir="ent_train_voz",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    learning_rate=5e-5,
    save_safetensors=False,
    overwrite_output_dir=True
)

data_collator = DataCollatorForTokenClassification(tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Обучение ===
trainer.train()
trainer.save_model("ent_train_voz/")
tokenizer.save_pretrained("ent_train_voz/")