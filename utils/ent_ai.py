import torch
from transformers import BertTokenizerFast, BertForTokenClassification

MODEL_DIR = "training/ent_train/ent_train_voz/"  # Папка с обученной моделью
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
    return parse_result(result)

def parse_result(result):
    data = {}

    for i in result:
        if i[1] == 'B-TYPE':
            if data.get('изделие') is None:
                data['изделие'] = i[0]
        elif i[1] == 'B-FSIZE':
            data['размер1'] = i[0]
        elif i[1] == 'B-SSIZE':
            data['размер2'] = i[0]
        elif i[1] == 'B-LENGTH':
            data['длинна'] = i[0]
        elif i[1] == 'B-MATERIAL':
            data['материал'] = i[0]
        elif i[1] == 'B-THICKNESS':
            data['толщина'] = i[0]
        elif i[1] == 'B-FLANGE':
            data['фланец'] = i[0]

    return data
