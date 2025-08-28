from transformers import BertTokenizerFast

MODEL_DIR = "training/ent_train/ent_train_voz/"  # Папка с обученной моделью
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

print(tokenizer.tokenize("Воздуховод круглый Ø100"))