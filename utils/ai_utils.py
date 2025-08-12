from sentence_transformers import SentenceTransformer

model = SentenceTransformer('training/labels_train/voz_pos_labels')
#paraphrase-multilingual-MiniLM-L12-v2

def get_embeddings(data, batch_size=32):
    embeddings = model.encode(data, batch_size=batch_size, show_progress_bar=True)
    return embeddings
