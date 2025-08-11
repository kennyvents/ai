import torch
from transformers import BertTokenizerFast, BertForTokenClassification

MODEL_DIR = "training/voz_pos/"

tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()

def extract_entities(text: str, to_lower: bool = True):
    s = text.lower() if to_lower else text  # если в датасете было в lower
    enc = tokenizer(s, return_tensors="pt", truncation=True, return_offsets_mapping=True)
    with torch.no_grad():
        logits = model(**{k:v for k,v in enc.items() if k in ("input_ids","attention_mask")}).logits

    pred_ids = logits.argmax(-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    offsets = enc["offset_mapping"][0].tolist()

    id2label = model.config.id2label
    print("id2label:", id2label)
    def to_label(i):  # у HF бывает dict со строковыми ключами
        return id2label[str(i)] if isinstance(id2label, dict) else id2label[i]

    result = []
    for tok, (start, end), pid in zip(tokens, offsets, pred_ids):
        # пропускаем спецтокены и продолжения wordpiece (##...)
        if tok in tokenizer.all_special_tokens:
            continue
        if tok.startswith("##"):
            continue
        result.append((tok, to_label(pid)))
    return result