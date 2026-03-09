# -*- coding: utf-8 -*-
"""
Vrew 리뷰 통합 분석 (2026년 3월)
1. 기존 + 신규 데이터 병합 & 중복 제거
2. 전처리 (텍스트 정제)
3. 감성 분석 (KoELECTRA)
4. 시각화 (별점 분포, 키워드 TOP30, 워드클라우드, 시계열, 플랫폼 비교)
"""

import csv
import os
import re
import sys
from collections import Counter
from pathlib import Path

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

CSV_DIR = BASE_DIR / "CSV 데이터"
IMG_DIR = BASE_DIR / "분석 이미지 파일"
SENT_DIR = BASE_DIR / "sentiment_out"
IMG_DIR.mkdir(parents=True, exist_ok=True)
SENT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis"
FONT_PATH = "/System/Library/Fonts/AppleGothic.ttf"

# ============================================================
# 불용어
# ============================================================
english_stopwords = set("""
the to of and in is it that for on with as are be was were at by this from
or but about not into up out over after so than then too can an no all would
there their what when which who how has had have will your more if my me do
just really very much also like use using used get got been only some
""".split())

korean_stopwords = set("""
그리고 하지만 그러나 그런 이런 저런 그냥 너무 정말 진짜 거의
근데 그래서 때문 이건 저건 그건 때 것 거 좀 더 등 듯
이번 다음 현재 오늘 어제 저희 우리 제가 저는
있습니다 좋습니다 합니다 됩니다 해요 되는 있어요 되요
좋아요 감사합니다 감사해요 대단히 정말로
같은 같이 처럼 보다 만큼 이나 라도 라서 니까
있는 있어서 같아요 많이 해도 하면 해야 다른 어떻게 계속
이거 다시 나옵니다 해주세요 없네요 그런데 에서
""".split())

stopwords = english_stopwords | korean_stopwords


# ============================================================
# 1. 데이터 병합 & 중복 제거
# ============================================================
def merge_and_deduplicate():
    print("=" * 60)
    print("STEP 1: 데이터 병합 & 중복 제거")
    print("=" * 60)

    old_path = CSV_DIR / "vrew_reviews_combined.csv"
    new_path = CSV_DIR / "vrew_reviews_combined_2026.csv"

    frames = []
    if old_path.exists():
        old = pd.read_csv(old_path)
        print(f"  기존 데이터: {len(old)}건")
        frames.append(old)
    if new_path.exists():
        new = pd.read_csv(new_path)
        print(f"  신규 데이터: {len(new)}건")
        frames.append(new)

    if not frames:
        print("  데이터가 없습니다!")
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True, sort=False)
    print(f"  병합 후: {len(df)}건")

    # 중복 제거 (App Store: review_id, Google Play: reviewId 기준)
    before = len(df)

    # App Store 중복 제거
    appstore_mask = df["platform"] == "appstore"
    appstore_df = df[appstore_mask].drop_duplicates(subset=["review_id", "country"], keep="last")

    # Google Play 중복 제거
    gplay_mask = df["platform"] == "googleplay"
    gplay_df = df[gplay_mask].drop_duplicates(subset=["reviewId", "country"], keep="last")

    df = pd.concat([appstore_df, gplay_df], ignore_index=True, sort=False)

    # content 기반 추가 중복 제거 (같은 플랫폼, 같은 국가, 같은 내용)
    df.drop_duplicates(subset=["platform", "country", "content"], keep="last", inplace=True)

    print(f"  중복 제거 후: {len(df)}건 (제거: {before - len(df)}건)")

    # 날짜 통합 컬럼 생성
    dates = []
    for _, row in df.iterrows():
        if row["platform"] == "appstore":
            dates.append(pd.to_datetime(row.get("updated"), errors="coerce", utc=True))
        else:
            dates.append(pd.to_datetime(row.get("at"), errors="coerce", utc=True))
    df["date"] = pd.to_datetime(dates, errors="coerce", utc=True)
    df["date"] = df["date"].dt.tz_localize(None)
    df.sort_values("date", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"  날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  플랫폼별: {df['platform'].value_counts().to_dict()}")
    print(f"  국가별: {df['country'].value_counts().to_dict()}")

    # review_text 컬럼 생성
    df["review_text"] = df["content"].fillna("").astype(str)

    # 저장
    save_path = CSV_DIR / "vrew_reviews_merged_2026.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  저장: {save_path}")
    print()
    return df


# ============================================================
# 2. 전처리
# ============================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    text = clean_text(text)
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", text)
    tokens = [t for t in tokens if t.lower() not in stopwords]
    return tokens


# ============================================================
# 3. 감성 분석
# ============================================================
def run_sentiment_analysis(df):
    print("=" * 60)
    print("STEP 2: 감성 분석 (KoELECTRA)")
    print("=" * 60)

    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"  device: {device}")
    print(f"  id2label: {model.config.id2label}")

    # 이 모델: 0=부정, 1=긍정
    pos_idx = 1

    texts = df["review_text"].fillna("").tolist()
    labels_kr = []
    pos_scores = []
    batch_size = 48

    for start in tqdm(range(0, len(texts), batch_size), desc="감성분석", unit="batch"):
        batch = [str(t) for t in texts[start:start + batch_size]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        for prob in probs:
            label_idx = int(prob.argmax())
            if label_idx == pos_idx:
                labels_kr.append("긍정")
            else:
                labels_kr.append("부정")
            pos_scores.append(float(prob[pos_idx]))

    df["Sentiment_label"] = labels_kr
    df["Sentiment_score"] = pos_scores

    print(f"\n  감성 분포:")
    print(f"    {df['Sentiment_label'].value_counts().to_dict()}")

    # 저장
    save_cols = ["platform", "country", "rating", "date", "review_text", "Sentiment_label", "Sentiment_score"]
    save_cols = [c for c in save_cols if c in df.columns]
    save_path = SENT_DIR / "reviews_with_sentiment_2026.csv"
    df[save_cols].to_csv(save_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"  저장: {save_path}")
    print()
    return df


# ============================================================
# 4. 시각화
# ============================================================
def visualize(df):
    print("=" * 60)
    print("STEP 3: 시각화")
    print("=" * 60)

    # --- 4-1. 별점 분포 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    rating_counts = df["rating"].value_counts().sort_index()
    colors = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]
    bars = ax.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(count),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("별점 분포", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("별점", fontsize=13)
    ax.set_ylabel("리뷰 수", fontsize=13)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "rating_distribution_2026.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  별점 분포 저장 완료")

    # --- 4-2. 감성별 키워드 TOP 30 ---
    pos_df = df[df["Sentiment_label"] == "긍정"]
    neg_df = df[df["Sentiment_label"] == "부정"]

    pos_words, neg_words = [], []
    for t in pos_df["review_text"]:
        pos_words.extend(tokenize(t))
    for t in neg_df["review_text"]:
        neg_words.extend(tokenize(t))

    pos_freq = Counter(pos_words).most_common(30)
    neg_freq = Counter(neg_words).most_common(30)

    print(f"\n  긍정 TOP 5: {pos_freq[:5]}")
    print(f"  부정 TOP 5: {neg_freq[:5]}")

    def plot_top(freq_data, title, output_file, color):
        if not freq_data:
            return
        words = [w for w, c in freq_data]
        counts = [c for w, c in freq_data]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(words[::-1], counts[::-1], color=color, edgecolor="white", linewidth=0.7)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("빈도수", fontsize=12)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    plot_top(pos_freq, "긍정 리뷰 단어 TOP 30", IMG_DIR / "pos_top30_2026.png", "#4CAF50")
    plot_top(neg_freq, "부정 리뷰 단어 TOP 30", IMG_DIR / "neg_top30_2026.png", "#F44336")
    print("  키워드 TOP30 저장 완료")

    # --- 4-3. 워드클라우드 ---
    try:
        from wordcloud import WordCloud

        def filter_stopwords(text):
            words = text.split()
            return " ".join(w for w in words if len(w) >= 2 and w.lower() not in stopwords)

        pos_text = filter_stopwords(" ".join(pos_df["review_text"].dropna().apply(clean_text)))
        neg_text = filter_stopwords(" ".join(neg_df["review_text"].dropna().apply(clean_text)))

        def make_wc(text, output, title, cmap):
            if not text.strip():
                return
            wc = WordCloud(
                font_path=FONT_PATH, width=1600, height=800,
                background_color="white", colormap=cmap,
                max_words=200, relative_scaling=0.3, min_font_size=10,
                stopwords=stopwords,
            ).generate(text)
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()

        make_wc(pos_text, IMG_DIR / "wordcloud_positive_2026.png", "긍정 리뷰 워드클라우드", "Greens")
        make_wc(neg_text, IMG_DIR / "wordcloud_negative_2026.png", "부정 리뷰 워드클라우드", "Reds")
        print("  워드클라우드 저장 완료")
    except ImportError:
        print("  wordcloud 패키지 없음 - 워드클라우드 건너뜀")

    # --- 4-4. 시계열: 월별 감성 추이 ---
    df_with_date = df.dropna(subset=["date"]).copy()
    df_with_date["year_month"] = df_with_date["date"].dt.to_period("M")

    monthly = df_with_date.groupby(["year_month", "Sentiment_label"]).size().unstack(fill_value=0)
    if "긍정" not in monthly.columns:
        monthly["긍정"] = 0
    if "부정" not in monthly.columns:
        monthly["부정"] = 0

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(monthly))
    labels = [str(p) for p in monthly.index]
    ax.bar([i - 0.2 for i in x], monthly["긍정"], width=0.4, color="#4CAF50", label="긍정", edgecolor="white")
    ax.bar([i + 0.2 for i in x], monthly["부정"], width=0.4, color="#F44336", label="부정", edgecolor="white")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("월별 감성 추이", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("리뷰 수", fontsize=12)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "monthly_sentiment_2026.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  월별 감성 추이 저장 완료")

    # --- 4-5. 플랫폼별 별점 비교 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, (platform, label) in enumerate([("appstore", "App Store"), ("googleplay", "Google Play")]):
        pf = df[df["platform"] == platform]
        rc = pf["rating"].value_counts().sort_index()
        axes[idx].bar(rc.index, rc.values, color=colors, edgecolor="white", linewidth=1.5)
        for bar_val, count in zip(rc.index, rc.values):
            axes[idx].text(bar_val, count + 1, str(count), ha="center", fontsize=10, fontweight="bold")
        axes[idx].set_title(f"{label} ({len(pf)}건)", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("별점", fontsize=12)
        axes[idx].set_xticks([1, 2, 3, 4, 5])
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
    axes[0].set_ylabel("리뷰 수", fontsize=12)
    plt.suptitle("플랫폼별 별점 비교", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "platform_comparison_2026.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  플랫폼별 비교 저장 완료")

    # --- 4-6. 국가별 감성 비율 ---
    country_sent = df.groupby(["country", "Sentiment_label"]).size().unstack(fill_value=0)
    if "긍정" not in country_sent.columns:
        country_sent["긍정"] = 0
    if "부정" not in country_sent.columns:
        country_sent["부정"] = 0
    country_sent["total"] = country_sent["긍정"] + country_sent["부정"]
    country_sent["긍정률"] = (country_sent["긍정"] / country_sent["total"] * 100).round(1)
    country_sent.sort_values("total", ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(country_sent))
    labels_c = country_sent.index.tolist()
    ax.bar([i - 0.2 for i in x], country_sent["긍정"], width=0.4, color="#4CAF50", label="긍정")
    ax.bar([i + 0.2 for i in x], country_sent["부정"], width=0.4, color="#F44336", label="부정")
    for i, (_, row) in enumerate(country_sent.iterrows()):
        ax.text(i, max(row["긍정"], row["부정"]) + 2, f"긍정률 {row['긍정률']}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_c, fontsize=12)
    ax.set_title("국가별 감성 분포", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("리뷰 수", fontsize=12)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "country_sentiment_2026.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  국가별 감성 분포 저장 완료")

    print()


# ============================================================
# 5. 요약 출력
# ============================================================
def print_summary(df):
    print("=" * 60)
    print("분석 요약")
    print("=" * 60)
    print(f"  총 리뷰 수: {len(df)}건")
    print(f"  날짜 범위: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  평균 별점: {df['rating'].mean():.2f}")
    print()
    print(f"  플랫폼별:")
    for p, cnt in df["platform"].value_counts().items():
        avg = df[df["platform"] == p]["rating"].mean()
        print(f"    {p}: {cnt}건 (평균 {avg:.2f}점)")
    print()
    print(f"  감성 분포:")
    for s, cnt in df["Sentiment_label"].value_counts().items():
        pct = cnt / len(df) * 100
        print(f"    {s}: {cnt}건 ({pct:.1f}%)")
    print()
    print(f"  생성된 파일:")
    print(f"    - CSV 데이터/vrew_reviews_merged_2026.csv")
    print(f"    - sentiment_out/reviews_with_sentiment_2026.csv")
    for f in sorted(IMG_DIR.glob("*_2026.png")):
        print(f"    - 분석 이미지 파일/{f.name}")
    print("=" * 60)


# ============================================================
# main
# ============================================================
def main():
    df = merge_and_deduplicate()
    df = run_sentiment_analysis(df)
    visualize(df)
    print_summary(df)
    print("완료!")


if __name__ == "__main__":
    main()
