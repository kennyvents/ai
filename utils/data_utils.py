import pandas as pd
import re


def normalize_name(name: str) -> str:
    name = re.sub(r'(?<=\d)x(?=\d)', '*', name)
    name = name.replace(',', '.')

    return name


def read_data(price, query):
    df_price = pd.read_excel(f'files/{price}.xlsx')
    df_query = pd.read_excel(f'files/{query}.xlsx', usecols=[0], header=None).iloc[:, 0]

    return df_price, df_query


def normalize_data(df_price, df_query):
    price_items = df_price['item'].fillna('').astype(str).apply(normalize_name).tolist()
    price_item_cost = df_price['cost'].tolist()

    query_items = df_query.fillna('').astype(str).apply(normalize_name).tolist()

    return price_items, query_items, price_item_cost

def get_result(result_data):
    rows = []
    for key, value in result_data.items():
        first = True
        for match, distance, cost in value:
            rows.append({
                'Позиции из запроса': key if first else '',
                'Позиции из прайса': match,
                'Сходство': float(distance),
                'Цена': cost
            })
            first = False

    df_result = pd.DataFrame(rows)
    df_result.to_excel(f'files/results/emb_results.xlsx', index=False)



