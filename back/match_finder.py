import numpy as np
from utils.ai_utils import get_embeddings

def find_top_matches(price, query, N=5):
    query_emb = get_embeddings(query)
    price_emb = get_embeddings(price, batch_size=64)

    q_vec = query_emb[0]

    norm_price_emb = price_emb / np.linalg.norm(price_emb, axis=1, keepdims=True)
    #т.е axis=1 - нормализуется каждая строка(массив) матрицы из 384d -> [x, y ,z ....384d]
    #keepdims - сохранение размерности нормализованных векторов

    q_vec_norm = q_vec / np.linalg.norm(q_vec)
    cos_similarities = np.dot(norm_price_emb, q_vec_norm)

    top_idx = np.argpartition(cos_similarities, -N)[-N:]
    top_idx = top_idx[np.argsort(-cos_similarities[top_idx])]

    return [(query[0],price[i], cos_similarities[i]) for i in top_idx]
