# -*- coding: utf-8 -*-
"""
CapCut 리뷰 크롤링 + 감성 분석 + 시각화
- Vrew와 동일한 파이프라인으로 분석하여 경쟁사 비교 가능하도록 구성
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
import requests
import time
from urllib.parse import urlparse, parse_qs
from google_play_scraper import reviews, Sort

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

CSV_DIR = BASE_DIR / "CSV 데이터"
IMG_DIR = BASE_DIR / "분석 이미지 파일"
SENT_DIR = BASE_DIR / "sentiment_out"
MODEL_NAME = "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis"
FONT_PATH = "/System/Library/Fonts/AppleGothic.ttf"

# CapCut 앱 ID
APPSTORE_ID = "1500855883"  # CapCut App Store ID
GPLAY_ID = "com.lemon.lvoverseas"  # CapCut Google Play ID

# 불용어
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
# 1. 크롤링
# ============================================================
def fetch_app_store_reviews(app_id, country="kr", max_pages=1000, sleep_sec=1.0):
    all_reviews = []
    consecutive_empty = 0

    print(f"[AppStore] CapCut 리뷰 수집 (country={country})")
    for page in range(1, max_pages + 1):
        url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                continue
            data = resp.json()
            if "feed" not in data or "entry" not in data["feed"]:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                continue
            entries = data["feed"]["entry"]
            count = 0
            for e in entries:
                if "im:rating" not in e:
                    continue
                count += 1
                all_reviews.append({
                    "platform": "appstore", "country": country,
                    "author": e.get("author", {}).get("name", {}).get("label", ""),
                    "title": e.get("title", {}).get("label", ""),
                    "content": e.get("content", {}).get("label", ""),
                    "rating": int(e["im:rating"]["label"]),
                    "version": e.get("im:version", {}).get("label", ""),
                    "updated": e.get("updated", {}).get("label", ""),
                    "review_id": e.get("id", {}).get("label", ""),
                })
            if count > 0:
                consecutive_empty = 0
                print(f"  page {page}: {count}개 (누적: {len(all_reviews)}개)")
            else:
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
            time.sleep(sleep_sec)
        except Exception as e:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                break
    print(f"[AppStore] {country} 총: {len(all_reviews)}개")
    return pd.DataFrame(all_reviews)


def fetch_google_play_reviews(app_id, lang="ko", country="kr", count_per_request=200, max_reviews=2000):
    all_reviews = []
    continuation_token = None
    req = 0

    print(f"[GooglePlay] CapCut 리뷰 수집 (lang={lang}, country={country}, max={max_reviews})")
    while True:
        req += 1
        try:
            result, continuation_token = reviews(
                app_id, lang=lang, country=country,
                sort=Sort.NEWEST, count=count_per_request,
                continuation_token=continuation_token
            )
            if not result:
                break
            all_reviews.extend(result)
            print(f"  요청 {req}: {len(result)}개 (누적: {len(all_reviews)}개)")
            if continuation_token is None or len(all_reviews) >= max_reviews:
                break
            time.sleep(0.5)
        except Exception as e:
            print(f"  요청 {req} 에러: {e}")
            break
    all_reviews = all_reviews[:max_reviews]

    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df.rename(columns={"userName": "author", "score": "rating"}, inplace=True)
        df["platform"] = "googleplay"
        df["lang"] = lang
        df["country"] = country
        if "reviewId" in df.columns:
            df.drop_duplicates(subset=["reviewId"], inplace=True)
    print(f"[GooglePlay] {lang}/{country} 총: {len(df)}개")
    return df


def crawl_capcut():
    print("=" * 60)
    print("STEP 1: CapCut 리뷰 크롤링")
    print("=" * 60)

    # App Store
    appstore_frames = []
    for country in ["kr", "us", "jp"]:
        df = fetch_app_store_reviews(APPSTORE_ID, country=country)
        if not df.empty:
            appstore_frames.append(df)
        print()
    appstore_df = pd.concat(appstore_frames, ignore_index=True) if appstore_frames else pd.DataFrame()

    # Google Play
    gplay_frames = []
    for locale in [{"lang": "ko", "country": "kr"}, {"lang": "en", "country": "us"}, {"lang": "ja", "country": "jp"}]:
        df = fetch_google_play_reviews(GPLAY_ID, lang=locale["lang"], country=locale["country"])
        if not df.empty:
            gplay_frames.append(df)
        print()
    gplay_df = pd.concat(gplay_frames, ignore_index=True) if gplay_frames else pd.DataFrame()

    # 병합
    combined = pd.concat([appstore_df, gplay_df], ignore_index=True, sort=False)
    combined.drop_duplicates(subset=["platform", "country", "content"], keep="last", inplace=True)

    # 날짜 통합
    dates = []
    for _, row in combined.iterrows():
        if row["platform"] == "appstore":
            dates.append(pd.to_datetime(row.get("updated"), errors="coerce", utc=True))
        else:
            dates.append(pd.to_datetime(row.get("at"), errors="coerce", utc=True))
    combined["date"] = pd.to_datetime(dates, errors="coerce", utc=True)
    combined["date"] = combined["date"].dt.tz_localize(None)
    combined.sort_values("date", ascending=False, inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined["review_text"] = combined["content"].fillna("").astype(str)

    save_path = CSV_DIR / "capcut_reviews_2026.csv"
    combined.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"\n  총 수집: {len(combined)}건")
    print(f"  날짜: {combined['date'].min()} ~ {combined['date'].max()}")
    print(f"  플랫폼: {combined['platform'].value_counts().to_dict()}")
    print(f"  국가: {combined['country'].value_counts().to_dict()}")
    print(f"  저장: {save_path}\n")
    return combined


# ============================================================
# 2. 감성 분석
# ============================================================
def run_sentiment(df):
    print("=" * 60)
    print("STEP 2: 감성 분석")
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

    pos_idx = 1
    texts = df["review_text"].fillna("").tolist()
    labels, scores = [], []

    for start in tqdm(range(0, len(texts), 48), desc="감성분석", unit="batch"):
        batch = [str(t) for t in texts[start:start + 48]]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1).cpu().numpy()
        for prob in probs:
            labels.append("긍정" if int(prob.argmax()) == pos_idx else "부정")
            scores.append(float(prob[pos_idx]))

    df["Sentiment_label"] = labels
    df["Sentiment_score"] = scores

    print(f"\n  감성 분포: {df['Sentiment_label'].value_counts().to_dict()}")

    save_cols = [c for c in ["platform", "country", "rating", "date", "review_text", "Sentiment_label", "Sentiment_score"] if c in df.columns]
    save_path = SENT_DIR / "capcut_with_sentiment_2026.csv"
    df[save_cols].to_csv(save_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    print(f"  저장: {save_path}\n")
    return df


# ============================================================
# 3. 시각화
# ============================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    text = clean_text(text)
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", text)
    return [t for t in tokens if t.lower() not in stopwords]


def visualize_capcut(df):
    print("=" * 60)
    print("STEP 3: 시각화")
    print("=" * 60)

    prefix = "capcut_"
    colors = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]

    # 별점 분포
    fig, ax = plt.subplots(figsize=(10, 5))
    rc = df["rating"].value_counts().sort_index()
    bars = ax.bar(rc.index, rc.values, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, rc.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(count),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("CapCut 별점 분포", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("별점", fontsize=13)
    ax.set_ylabel("리뷰 수", fontsize=13)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"{prefix}rating_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  별점 분포 저장")

    # 감성별 키워드 TOP 30
    pos_df = df[df["Sentiment_label"] == "긍정"]
    neg_df = df[df["Sentiment_label"] == "부정"]

    pos_words, neg_words = [], []
    for t in pos_df["review_text"]:
        pos_words.extend(tokenize(t))
    for t in neg_df["review_text"]:
        neg_words.extend(tokenize(t))

    pos_freq = Counter(pos_words).most_common(30)
    neg_freq = Counter(neg_words).most_common(30)

    print(f"  긍정 TOP 5: {pos_freq[:5]}")
    print(f"  부정 TOP 5: {neg_freq[:5]}")

    def plot_top(freq, title, output, color):
        if not freq:
            return
        words = [w for w, c in freq]
        counts = [c for w, c in freq]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(words[::-1], counts[::-1], color=color, edgecolor="white")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("빈도수", fontsize=12)
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()

    plot_top(pos_freq, "CapCut 긍정 리뷰 단어 TOP 30", IMG_DIR / f"{prefix}pos_top30.png", "#4CAF50")
    plot_top(neg_freq, "CapCut 부정 리뷰 단어 TOP 30", IMG_DIR / f"{prefix}neg_top30.png", "#F44336")
    print("  키워드 TOP30 저장")

    # 워드클라우드
    try:
        from wordcloud import WordCloud

        def filter_sw(text):
            return " ".join(w for w in text.split() if len(w) >= 2 and w.lower() not in stopwords)

        pos_text = filter_sw(" ".join(pos_df["review_text"].dropna().apply(clean_text)))
        neg_text = filter_sw(" ".join(neg_df["review_text"].dropna().apply(clean_text)))

        def make_wc(text, output, title, cmap):
            if not text.strip():
                return
            wc = WordCloud(font_path=FONT_PATH, width=1600, height=800,
                          background_color="white", colormap=cmap, max_words=200,
                          relative_scaling=0.3, min_font_size=10, stopwords=stopwords).generate(text)
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()

        make_wc(pos_text, IMG_DIR / f"{prefix}wordcloud_positive.png", "CapCut 긍정 워드클라우드", "Greens")
        make_wc(neg_text, IMG_DIR / f"{prefix}wordcloud_negative.png", "CapCut 부정 워드클라우드", "Reds")
        print("  워드클라우드 저장")
    except ImportError:
        print("  wordcloud 없음 - 건너뜀")

    # 플랫폼별 비교
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for idx, (p, label) in enumerate([("appstore", "App Store"), ("googleplay", "Google Play")]):
        pf = df[df["platform"] == p]
        rc = pf["rating"].value_counts().sort_index()
        axes[idx].bar(rc.index, rc.values, color=colors, edgecolor="white", linewidth=1.5)
        for r, c in zip(rc.index, rc.values):
            axes[idx].text(r, c + 1, str(c), ha="center", fontsize=10, fontweight="bold")
        axes[idx].set_title(f"CapCut {label} ({len(pf)}건)", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("별점", fontsize=12)
        axes[idx].set_xticks([1, 2, 3, 4, 5])
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
    axes[0].set_ylabel("리뷰 수", fontsize=12)
    plt.suptitle("CapCut 플랫폼별 별점 비교", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"{prefix}platform_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  플랫폼별 비교 저장")

    # 국가별 감성
    cs = df.groupby(["country", "Sentiment_label"]).size().unstack(fill_value=0)
    for col in ["긍정", "부정"]:
        if col not in cs.columns:
            cs[col] = 0
    cs["total"] = cs["긍정"] + cs["부정"]
    cs["긍정률"] = (cs["긍정"] / cs["total"] * 100).round(1)
    cs.sort_values("total", ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(cs))
    ax.bar([i - 0.2 for i in x], cs["긍정"], width=0.4, color="#4CAF50", label="긍정")
    ax.bar([i + 0.2 for i in x], cs["부정"], width=0.4, color="#F44336", label="부정")
    for i, (_, row) in enumerate(cs.iterrows()):
        ax.text(i, max(row["긍정"], row["부정"]) + 2, f"긍정률 {row['긍정률']}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cs.index.tolist(), fontsize=12)
    ax.set_title("CapCut 국가별 감성 분포", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("리뷰 수", fontsize=12)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"{prefix}country_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  국가별 감성 저장")

    print()


# ============================================================
# 4. Vrew vs CapCut 비교 시각화
# ============================================================
def compare_with_vrew(capcut_df):
    print("=" * 60)
    print("STEP 4: Vrew vs CapCut 비교")
    print("=" * 60)

    vrew_sent = pd.read_csv(SENT_DIR / "reviews_with_sentiment_2026.csv")

    # 기본 통계 출력
    print(f"\n  {'항목':<20} {'Vrew':<18} {'CapCut':<18}")
    print(f"  {'-'*56}")
    print(f"  {'총 리뷰 수':<20} {len(vrew_sent):<18} {len(capcut_df):<18}")
    print(f"  {'평균 별점':<20} {vrew_sent['rating'].mean():<18.2f} {capcut_df['rating'].mean():<18.2f}")

    vrew_pos = (vrew_sent['Sentiment_label'] == '긍정').sum()
    cap_pos = (capcut_df['Sentiment_label'] == '긍정').sum()
    print(f"  {'긍정률':<20} {vrew_pos/len(vrew_sent)*100:<18.1f} {cap_pos/len(capcut_df)*100:<18.1f}")

    colors_app = ["#F44336", "#FF9800", "#FFC107", "#8BC34A", "#4CAF50"]

    # 별점 분포 비교
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for idx, (data, name) in enumerate([(vrew_sent, f"Vrew ({len(vrew_sent)}건)"), (capcut_df, f"CapCut ({len(capcut_df)}건)")]):
        rc = data["rating"].value_counts().sort_index()
        bars = axes[idx].bar(rc.index, rc.values, color=colors_app, edgecolor="white", linewidth=1.5)
        for bar, count in zip(bars, rc.values):
            pct = count / len(data) * 100
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rc.values)*0.02,
                          f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        axes[idx].set_title(name, fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("별점", fontsize=12)
        axes[idx].set_xticks([1, 2, 3, 4, 5])
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
    axes[0].set_ylabel("리뷰 수", fontsize=12)
    plt.suptitle("Vrew vs CapCut 별점 분포", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "vs_rating_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  별점 비교 저장")

    # 감성 비교
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(2)
    width = 0.35
    vrew_vals = [(vrew_sent['Sentiment_label'] == '긍정').sum(), (vrew_sent['Sentiment_label'] == '부정').sum()]
    cap_vals = [(capcut_df['Sentiment_label'] == '긍정').sum(), (capcut_df['Sentiment_label'] == '부정').sum()]

    b1 = ax.bar(x - width/2, vrew_vals, width, label='Vrew', color=['#81C784', '#E57373'], edgecolor='white')
    b2 = ax.bar(x + width/2, cap_vals, width, label='CapCut', color=['#4CAF50', '#F44336'], edgecolor='white')

    for bars, total in [(b1, len(vrew_sent)), (b2, len(capcut_df))]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 3, f"{int(h)}\n({h/total*100:.1f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["긍정", "부정"], fontsize=13)
    ax.set_title("Vrew vs CapCut 감성 분포", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylabel("리뷰 수", fontsize=12)
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "vs_sentiment.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  감성 비교 저장")

    # 부정 키워드 비교
    def get_neg_keywords(data):
        words = []
        for t in data[data["Sentiment_label"] == "부정"]["review_text"]:
            words.extend(tokenize(t))
        return Counter(words).most_common(15)

    vrew_neg = get_neg_keywords(vrew_sent)
    cap_neg = get_neg_keywords(capcut_df)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for idx, (freq, title, color) in enumerate([
        (vrew_neg, "Vrew 부정 키워드 TOP 15", "#FF7043"),
        (cap_neg, "CapCut 부정 키워드 TOP 15", "#E53935"),
    ]):
        if not freq:
            continue
        words = [w for w, c in freq]
        counts = [c for w, c in freq]
        axes[idx].barh(words[::-1], counts[::-1], color=color, edgecolor="white")
        axes[idx].set_title(title, fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("빈도수", fontsize=11)
        axes[idx].grid(axis="x", alpha=0.3, linestyle="--")
        axes[idx].spines["top"].set_visible(False)
        axes[idx].spines["right"].set_visible(False)
    plt.suptitle("부정 키워드 비교: Vrew vs CapCut", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "vs_negative_keywords.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  부정 키워드 비교 저장")

    print()


# ============================================================
# 5. 요약
# ============================================================
def print_summary(df):
    print("=" * 60)
    print("CapCut 분석 요약")
    print("=" * 60)
    print(f"  총 리뷰: {len(df)}건")
    if df['date'].notna().any():
        print(f"  기간: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  평균 별점: {df['rating'].mean():.2f}")
    print(f"\n  플랫폼별:")
    for p, cnt in df["platform"].value_counts().items():
        avg = df[df["platform"] == p]["rating"].mean()
        print(f"    {p}: {cnt}건 (평균 {avg:.2f}점)")
    print(f"\n  감성 분포:")
    for s, cnt in df["Sentiment_label"].value_counts().items():
        print(f"    {s}: {cnt}건 ({cnt/len(df)*100:.1f}%)")
    print("=" * 60)


def main():
    df = crawl_capcut()
    df = run_sentiment(df)
    visualize_capcut(df)
    compare_with_vrew(df)
    print_summary(df)
    print("완료!")


if __name__ == "__main__":
    main()
