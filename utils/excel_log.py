import pandas as pd
from utils.ent_ai import extract_entities

def get_top_ent(input_filename, output_filename, N=2):
    df = pd.read_excel(f'{input_filename}.xlsx',
    header = None, names=(['key', 'value', 'cos_sim', 'cost']), skiprows=1)

    data = {}
    current_key = None
    current_key_dict = {}

    for idx, row in df.iterrows():
        if pd.notna(row['key']):
            current_key = row['key'].lower()
            current_key_dict = extract_entities(current_key)
            data[current_key] = []

        value = row['value'].lower()
        cost = row['cost']

        value_ent = extract_entities(value)

        product = (current_key_dict.get('изделие'), value_ent.get('изделие'))
        size1 = (int(current_key_dict.get('размер1', 0)), int(value_ent.get('размер1', 0)))
        size2 = (int(current_key_dict.get('размер2', 0)), int(value_ent.get('размер2', 0)))
        length = (int(current_key_dict.get('длинна', 0)), int(value_ent.get('длинна', 0)))

        condition_prod = (product[0] == product[1])
        condition_size1 = ((size1[1] in range(size1[0], size1[0] + int(size1[0] * 0.3) + 1, 10)) or (size1[1] in range(size1[0], size1[0] - int(size1[0] * 0.3) - 1, -10)))
        condition_size2 = ((size2[1] in range(size2[0], size2[0] + int(size2[0] * 0.3) + 1, 10)) or (size2[1] in range(size2[0], size2[0] - int(size2[0] * 0.3) - 1, -10)))
        condition_length = (length[1] in range(length[0], length[0] + int(length[0] * 0.3) + 1, 10) or length[1] in range(length[0], length[0] - int(length[0] * 0.3) - 1, -10))

        if condition_prod and condition_size1 and condition_size2 and condition_length:
            data[current_key].append((value, cost))

        # print(current_key, value)
        # print(condition_prod)
        # print(product)
        # print()
        #
        # print(condition_size1,condition_size2,condition_length)
        # print()

    rows = []
    for key, value in data.items():
        print(key, value)
        if len(value) >= N:
            rows.append([key, value[0][0], value[0][1]])
            rows.append(['', value[1][0], value[1][1]])
        elif len(value) == 1:
            rows.append([key, value[0][0], value[0][1]])
        else:
            rows.append([key, 'не найден'])

    output_df = pd.DataFrame(data=rows, columns=['запрос', 'позиция прайса', 'цена'])
    output_df.to_excel(f'{output_filename}.xlsx', index=False)

