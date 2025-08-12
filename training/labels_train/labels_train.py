# -*- coding: utf-8 -*-
"""
Fine-tune paraphrase-multilingual-MiniLM-L12-v2 from an Excel sheet with labels.
Expected Excel schema (on one sheet):
    text1 | text2 | label   where label in [0..1] (e.g., 0, 0.5, 1)

Usage:
    pip install -U sentence-transformers pandas scikit-learn openpyxl torch
    python train_from_excel_labels.py

If you want to change file/sheet/params, edit the CFG section below.
"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


# =======================
# Конфиг по умолчанию
# =======================
@dataclass
class CFG:
    # I/O
    excel_path: str = "voz_train.xlsx"
    sheet_with_labels: str = "Sheet1"   # лист с колонками: text1, text2, label
    output_dir: str = "voz_pos_labels"

    # Колонки (если в файле другие названия — поменяй тут)
    col_text1: str = "text1"
    col_text2: str = "text2"
    col_label: str = "label"

    # Модель/оптимизация
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 64
    num_epochs: int = 3
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    fp16: bool = True
    eval_steps: int = 500
    save_best: bool = True

    # Препроцесс
    lowercase: bool = False       # привести тексты к нижнему регистру
    strip_spaces: bool = True     # схлопнуть лишние пробелы
    clip_labels01: bool = True    # обрезать метку в [0,1]

    # Сплит/баланс
    test_size: float = 0.1
    random_state: int = 42
    balance_classes: bool = False  # включить простой оверсэмплинг классов

    # Лог
    show_progress_bar: bool = True


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_text(s: str, lowercase: bool, strip_spaces: bool) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    if strip_spaces:
        s = " ".join(s.split())
    if lowercase:
        s = s.lower()
    return s


def load_with_labels(path: str, sheet: str, col_text1: str, col_text2: str, col_label: str,
                     lowercase: bool, strip_spaces: bool, clip_labels01: bool,
                     drop_dups: bool = True) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

    # поддержка произвольных имён колонок (регистронезависимо)
    cols_lower = {c.lower(): c for c in df.columns}
    t1 = cols_lower.get(col_text1.lower(), col_text1)
    t2 = cols_lower.get(col_text2.lower(), col_text2)
    lb = cols_lower.get(col_label.lower(), col_label)

    for col in [t1, t2, lb]:
        if col not in df.columns:
            raise ValueError(f"На листе '{sheet}' нет колонки '{col}'")

    df = df[[t1, t2, lb]].rename(columns={t1: "text1", t2: "text2", lb: "label"})

    # чистка
    df["text1"] = df["text1"].map(lambda s: clean_text(s, lowercase, strip_spaces))
    df["text2"] = df["text2"].map(lambda s: clean_text(s, lowercase, strip_spaces))
    df = df[(df["text1"] != "") & (df["text2"] != "")]

    # метки -> float
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    if clip_labels01:
        df["label"] = df["label"].clip(0.0, 1.0)

    if drop_dups:
        df = df.drop_duplicates(subset=["text1", "text2", "label"])

    df = df.reset_index(drop=True)
    return df


def oversample_by_label(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Простой оверсэмплинг до размера самого большого класса."""
    vc = df["label"].value_counts()
    # если непрерывные метки, оверсэмплинг не имеет смысла — пропускаем
    if len(vc) > 10:
        print("⚠️ Много уникальных меток — пропускаю балансировку.")
        return df

    target = int(vc.max())
    parts = []
    for val, group in df.groupby("label"):
        k = target // len(group)
        r = target % len(group)
        chunk = pd.concat([group] * k + [group.sample(r, replace=True, random_state=seed)])
        parts.append(chunk)
    df_bal = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)
    return df_bal.reset_index(drop=True)


def build_examples(frame: pd.DataFrame):
    return [
        InputExample(texts=[r["text1"], r["text2"]], label=float(r["label"]))
        for _, r in frame.iterrows()
    ]


def train(cfg: CFG):
    set_global_seed(cfg.random_state)

    print(f"📄 Reading: {cfg.excel_path} [{cfg.sheet_with_labels}]")
    df = load_with_labels(
        path=cfg.excel_path,
        sheet=cfg.sheet_with_labels,
        col_text1=cfg.col_text1,
        col_text2=cfg.col_text2,
        col_label=cfg.col_label,
        lowercase=cfg.lowercase,
        strip_spaces=cfg.strip_spaces,
        clip_labels01=cfg.clip_labels01,
    )

    print(f"✅ Loaded {len(df)} pairs. Label distribution:")
    print(df["label"].value_counts(dropna=False).sort_index())

    # стратификация, если дискретные метки
    stratify_col: Optional[pd.Series] = None
    if df["label"].nunique() <= 10:  # эвристика
        stratify_col = df["label"]
    else:
        print("ℹ️ Метка выглядит непрерывной — стратификацию отключаю.")

    # балансировка (по желанию)
    if cfg.balance_classes:
        df = oversample_by_label(df, seed=cfg.random_state)
        print("🔁 After balancing:")
        print(df["label"].value_counts(dropna=False).sort_index())

    train_df, dev_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        shuffle=True,
        stratify=stratify_col if stratify_col is not None else None,
    )

    print(f"🧪 Train: {len(train_df)} | Dev: {len(dev_df)}")

    train_examples = build_examples(train_df)
    dev_examples = build_examples(dev_df)

    model = SentenceTransformer(cfg.model_name)

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=cfg.batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name="dev")

    warmup_steps = math.ceil(len(train_loader) * cfg.num_epochs * cfg.warmup_ratio)

    print("🚀 Training started...")
    model.fit(
        train_objectives=[(train_loader, train_loss)],
        evaluator=evaluator,
        epochs=cfg.num_epochs,
        evaluation_steps=cfg.eval_steps,
        warmup_steps=warmup_steps,
        output_path=cfg.output_dir,
        optimizer_params={"lr": cfg.lr},
        use_amp=cfg.fp16,
        save_best_model=cfg.save_best,
        show_progress_bar=cfg.show_progress_bar,
    )
    print("✅ Training finished.")
    print(f"💾 Best (or last) model saved to: {cfg.output_dir}")

    # Быстрая проверка
    best = SentenceTransformer(cfg.output_dir)
    s1 = "воздуховод 300x150 оц 0.7"
    s2 = "круглый воздуховод 300 мм оцинкованный"
    emb = best.encode([s1, s2], convert_to_tensor=True, normalize_embeddings=True)
    sim = util.cos_sim(emb[0], emb[1]).item()
    print(f"🔍 Cosine('{s1}', '{s2}') = {sim:.4f}")


def parse_args_to_cfg() -> CFG:
    parser = argparse.ArgumentParser(description="Fine-tune pmMiniLM from Excel with labels.")
    parser.add_argument("--excel", type=str, default=None, help="Path to .xlsx with text1,text2,label")
    parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--balance", action="store_true", help="Enable simple class balancing")
    parser.add_argument("--lower", action="store_true", help="Lowercase all texts")

    args = parser.parse_args()

    cfg = CFG()
    if args.excel is not None:
        cfg.excel_path = args.excel
    if args.sheet is not None:
        cfg.sheet_with_labels = args.sheet
    if args.out is not None:
        cfg.output_dir = args.out
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.lr is not None:
        cfg.lr = args.lr
    if args.eval_steps is not None:
        cfg.eval_steps = args.eval_steps
    if args.balance:
        cfg.balance_classes = True
    if args.lower:
        cfg.lowercase = True

    return cfg


if __name__ == "__main__":
    cfg = parse_args_to_cfg()
    train(cfg)
