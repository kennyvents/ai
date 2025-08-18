import numpy as np
from utils.ai_utils import get_embeddings

def find_top_matches(price, query, cost, N=5):
    print(query)
    top_dict = {}

    query_embs = get_embeddings(query)
    price_embs = get_embeddings(price, batch_size=64)

    norm_price_emb = price_embs / np.linalg.norm(price_embs, axis=1, keepdims=True)
    # т.е axis=1 - нормализуется каждая строка(массив) матрицы из 384d -> [x, y ,z ....384d]
    # keepdims - сохранение размерности нормализованных векторов

    query_idx = 0

    for query_emb in query_embs:


        q_vec = query_emb
        q_vec_norm = q_vec / np.linalg.norm(q_vec)

        cos_similarities = np.dot(norm_price_emb, q_vec_norm)

        top_idx = np.argpartition(cos_similarities, -N)[-N:]
        top_idx = top_idx[np.argsort(-cos_similarities[top_idx])]
        print(top_idx)

        for i in top_idx:
            top_dict.setdefault(query[query_idx], []).append((price[i], cos_similarities[i], cost[i]))

        query_idx += 1

    return top_dict
