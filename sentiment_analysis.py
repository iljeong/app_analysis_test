#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Vrew 리뷰 감정분석 스크립트
- 입력: 전처리 완료 CSV (브류 리뷰 뜯어보기.py에서 생성된 vrew_reviews_tokens.csv)
- 모델: jaehyeong/koelectra-base-v3-generalized-sentiment-analysis
- 출력: sentiment_out/reviews_with_sentiment.csv
"""

import csv
import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path("/Users/seojeong-il/Desktop/내문서/데이터 분석/개인 분석/보이저엑스/vrew")
CACHE_DIR = BASE_DIR / ".cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(HF_CACHE_DIR / "transformers").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR / "transformers"))
os.environ.setdefault("HF_HUB_ENABLE_HF_XET", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# 1. 경로 및 기본 설정
# ============================================================
INPUT_PATH = BASE_DIR / "vrew_reviews_tokens.csv"
OUT_DIR = BASE_DIR / "sentiment_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUT_DIR / "reviews_with_sentiment.csv"

MODEL_NAME = "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis"


# ============================================================
# 2. 데이터 로드
# ============================================================
def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")

    df = pd.read_csv(path)
    if "review_text" not in df.columns:
        if "content" in df.columns:
            df["review_text"] = df["content"].astype(str)
        else:
            raise ValueError("CSV에 'review_text' 컬럼이 필요합니다.")
    return df


# ============================================================
# 3. 모델 및 토크나이저 준비
# ============================================================
def load_model_and_tokenizer():
    print(f"[INFO] 모델 로드: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"[INFO] device = {device}")
    print("[INFO] id2label:", model.config.id2label)
    return tokenizer, model, device


def resolve_label_indices(model) -> tuple[int, int]:
    id2label = model.config.id2label
    pos_idx = None
    neg_idx = None

    for idx, label in id2label.items():
        lowered = label.lower()
        if "pos" in lowered or "긍정" in lowered:
            pos_idx = idx
        elif "neg" in lowered or "부정" in lowered:
            neg_idx = idx

    if pos_idx is None:
        pos_idx = 1
    if neg_idx is None:
        neg_idx = 0

    print(f"[INFO] label index - neg:{neg_idx}, pos:{pos_idx}")
    return neg_idx, pos_idx


def map_label(idx: int, id2label: dict[str, str], pos_idx: int) -> str:
    raw = id2label.get(idx, "").lower()
    if "pos" in raw or "긍정" in raw:
        return "긍정"
    if "neg" in raw or "부정" in raw:
        return "부정"
    return "긍정" if idx == pos_idx else "부정"


# ============================================================
# 4. 배치 추론
# ============================================================
def predict_batch(
    texts: list[str],
    tokenizer,
    model,
    device,
    pos_idx: int,
    batch_size: int = 64,
    max_len: int = 128,
) -> tuple[list[str], list[float]]:
    labels_kr: list[str] = []
    pos_scores: list[float] = []
    id2label = model.config.id2label

    for start in tqdm(range(0, len(texts), batch_size), desc="predict", unit="batch"):
        batch = list(map(str, texts[start:start + batch_size]))

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for prob in probs:
            label_idx = int(prob.argmax())
            labels_kr.append(map_label(label_idx, id2label, pos_idx))
            pos_scores.append(float(prob[pos_idx]))

    return labels_kr, pos_scores


# ============================================================
# 5. 후처리 및 저장
# ============================================================
def sanitize_text(text: str) -> str:
    sanitized = "" if pd.isna(text) else str(text)
    return (
        sanitized.replace("\r\n", " ")
        .replace("\n", " ")
        .replace("\t", " ")
        .replace("\x00", "")
    )


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    for column in df.columns:
        if "date" in column.lower():
            return column
    return None


def save_with_sentiment(df: pd.DataFrame, date_col: Optional[str], path: Path):
    save_cols: list[str] = []
    candidate_cols = [
        "ID",
        "provider",
        "store",
        "rating",
        date_col,
        "review_text",
        "Sentiment_label",
        "Sentiment_score",
    ]

    for col in candidate_cols:
        if col and col in df.columns:
            save_cols.append(col)

    df[save_cols].to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    print(f"[INFO] 감정분석 결과 저장 → {path}")


# ============================================================
# main
# ============================================================
def main():
    df = load_dataframe(INPUT_PATH)
    print(f"[INFO] 입력 데이터: {len(df):,}건")

    date_col = detect_date_column(df)
    print(f"[INFO] 날짜 컬럼: {date_col}")

    tokenizer, model, device = load_model_and_tokenizer()
    neg_idx, pos_idx = resolve_label_indices(model)

    texts = df["review_text"].fillna("").tolist()
    print("[INFO] 감정분석 시작...")
    labels, scores = predict_batch(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        pos_idx=pos_idx,
        batch_size=48,
        max_len=128,
    )

    df["Sentiment_label"] = labels
    df["Sentiment_score"] = scores
    print("[INFO] 감정분석 샘플:")
    print(df[["review_text", "Sentiment_label", "Sentiment_score"]].head().to_string(index=False))

    df["review_text"] = df["review_text"].map(sanitize_text)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)

    save_with_sentiment(df, date_col, OUTPUT_PATH)


if __name__ == "__main__":
    main()

