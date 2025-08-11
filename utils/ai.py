import torch
from transformers import BertTokenizerFast, BertForTokenClassification

MODEL_DIR = "training/voz_pos/"  # Папка с обученной моделью
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForTokenClassification.from_pretrained(MODEL_DIR)
model.eval()


def extract_entities(text: str, to_lower: bool = True):
    s = text.lower() if to_lower else text  # если в датасете было в lowercase
    enc = tokenizer(s, return_tensors="pt", truncation=True, return_offsets_mapping=True)

    with torch.no_grad():
        logits = model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits

    pred_ids = logits.argmax(-1)[0].tolist()  # Предсказания модели
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    offsets = enc["offset_mapping"][0].tolist()

    id2label = model.config.id2label
    print("id2label:", id2label)  # Проверим на всех доступных метках

    # Функция для перевода ID в метку с проверкой
    def to_label(i):
        if i not in id2label:
            print(f"Warning: ID {i} not found in id2label. Using 'O'.")
            return "O"
        return id2label[i]  # Если id2label — это словарь, используем его как есть

    result = []
    for tok, (start, end), pid in zip(tokens, offsets, pred_ids):
        # Пропускаем спецтокены и продолжения wordpiece (например, "##")
        if tok in tokenizer.all_special_tokens:
            continue
        if tok.startswith("##"):
            continue
        result.append((tok, to_label(pid)))  # Получаем метку из id2label
    return result
