import pandas as pd
import re
from pathlib import Path

def transform_name(s: str) -> str:
    if not isinstance(s, str):
        return s
    text = s.strip()

    # Если уже есть "круглый" или "прямоугольный", пропускаем
    if re.search(r'(?i)\bвоздуховод\b', text) and re.search(r'(?i)\b(круглый|прямоугольный)\b', text):
        return text

    # Нормализуем разделители размеров: x/х/Х → *
    text_norm = text.replace('Х', '*').replace('х', '*').replace('x', '*')
    parts = [p.strip() for p in text_norm.split('-')]

    if len(parts) < 2:
        return text  # нет размеров

    # Первый элемент должен начинаться с "воздуховод"
    if not re.match(r'(?i)^воздуховод\b', parts[0]):
        return text

    size_token = parts[1] if len(parts) >= 2 else ""

    shape = None
    # Два размера через *
    if re.match(r'^\d{2,4}\*\d{2,4}$', size_token):
        shape = "прямоугольный"
    # Один размер
    elif re.match(r'^\d{2,4}$', size_token):
        shape = "круглый"
    # Один размер с диаметром
    elif re.match(r'^[Ø∅]?(\d{2,4})$', size_token):
        shape = "круглый"

    if shape is None:
        return text

    parts[0] = "Воздуховод " + shape
    return "-".join(parts)

def process_excel(src_path: str, dst_path: str):
    df = pd.read_excel(src_path)
    if df.shape[1] == 0:
        raise RuntimeError("Файл без столбцов. В первом столбце должны быть наименования.")

    first_col = df.columns[0]
    df[first_col] = df[first_col].apply(transform_name)
    df.to_excel(dst_path, index=False)
    print(f"Готово: {dst_path}")

if __name__ == "__main__":
    src = "catalog_data.xlsx"           # исходный файл
    dst = "modified_prices.xlsx"    # файл с изменениями
    process_excel(src, dst)
