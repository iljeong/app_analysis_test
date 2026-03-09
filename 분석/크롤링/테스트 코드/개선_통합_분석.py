"""
개선된 통합 분석 스크립트
========================
한계점 개선 사항:
1. kiwipiepy 형태소 분석기 (한국어 lemmatization) — "자막이/자막을/자막" → "자막"
2. langdetect 언어 감지 — 비한국어 리뷰 정확도 플래그
3. 3클래스 감성 분석 (긍정/중립/부정) — threshold 기반
4. 도메인 특화 불용어 — 앱 이름, 일반 표현 등 추가
5. LDA 토픽 모델링 — 긍정/부정 각각 주요 토픽 추출
6. 감성 모델 검증 — 별점 기반 기대값과 비교, confusion matrix
7. max_length 256으로 확대 — 긴 리뷰 잘림 최소화
8. 개선된 시각화 — 토픽, 검증 결과 포함

실행: python 개선_통합_분석.py [vrew|capcut|compare|all]
"""

import re
import csv
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 경로 설정
# ============================================================
BASE = Path(__file__).resolve().parent.parent.parent.parent
CSV_DIR = BASE / "CSV 데이터"
SENT_DIR = BASE / "sentiment_out"
IMG_DIR = BASE / "분석 이미지 파일"
SENT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# 폰트
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

MODEL_NAME = "jaehyeong/koelectra-base-v3-generalized-sentiment-analysis"

# ============================================================
# 1. 불용어 (개선: 도메인 특화 추가)
# ============================================================
english_stopwords = set("""
the to of and in is it that for on with as are be was were at by this from
or but about not into up out over after so than then too can an no all would
there their what when which who how has had have will your more if my me do
just really very much also like use using used get got been only some
great good nice well its been does did don doesn didn doesn
""".split())

korean_stopwords = set("""
그리고 하지만 그러나 그런 이런 저런 그냥 너무 정말 진짜 거의
근데 그래서 때문 이건 저건 그건 때 것 거 좀 더 등 듯
이번 다음 현재 오늘 어제 저희 우리 제가 저는
있습니다 좋습니다 합니다 됩니다 해요 되는 있어요 되요 돼요
좋아요 감사합니다 감사해요 대단히 정말로
같은 같이 처럼 보다 만큼 이나 라도 라서 니까
있는 있어서 같아요 많이 해도 하면 해야 다른 어떻게 계속
이거 다시 나옵니다 해주세요 없네요 그런데 에서
하는 되는 되어 있는 없는 하고 되고 인데 으로 에서
입니다 습니다 는데 이요 어요 아요
그래 이제 아니 그게 이게 저게 여기 거기 저기
""".split())

japanese_stopwords = set("""
です ます した して いる ある この その あの ない なる れる
ている ました ません から まで ので ため よう もの こと
""".split())

domain_stopwords_vrew = set("""
vrew 브루 브류 Vrew VREW
""".split())

domain_stopwords_capcut = set("""
capcut CapCut 캡컷 CAPCUT
""".split())

common_domain_stopwords = set("""
앱 어플 프로그램 어플리케이션 app application
""".split())

stopwords_vrew = english_stopwords | korean_stopwords | japanese_stopwords | domain_stopwords_vrew | common_domain_stopwords
stopwords_capcut = english_stopwords | korean_stopwords | japanese_stopwords | domain_stopwords_capcut | common_domain_stopwords

# ============================================================
# 2. kiwipiepy 형태소 분석기
# ============================================================
from kiwipiepy import Kiwi
kiwi = Kiwi()

# 한국어 형태소 분석으로 토큰화 (명사, 동사 어간, 형용사 어간, 영어)
VALID_TAGS = {"NNG", "NNP", "NNB", "VV", "VA", "SL"}

def tokenize_korean(text, sw):
    """kiwipiepy 기반 한국어 형태소 분석 + 원형 추출"""
    if not text or pd.isna(text):
        return []
    text = str(text)
    try:
        result = kiwi.tokenize(text)
    except Exception:
        return []
    tokens = []
    for tok in result:
        form = tok.form
        tag = tok.tag
        if tag in VALID_TAGS and len(form) >= 2 and form.lower() not in sw:
            tokens.append(form)
    return tokens


def tokenize_english(text, sw):
    """영문 정규식 기반 토큰화"""
    if not text or pd.isna(text):
        return []
    text = re.sub(r"[^A-Za-z\s]", " ", str(text))
    tokens = re.findall(r"[A-Za-z]{2,}", text)
    return [t.lower() for t in tokens if t.lower() not in sw]


def tokenize_japanese(text, sw):
    """일본어 — kiwipiepy가 지원하지 않으므로 간단한 정규식 사용"""
    if not text or pd.isna(text):
        return []
    text = re.sub(r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uAC00-\uD7AFa-zA-Z\s]", " ", str(text))
    # 한자·카타카나·히라가나 2글자 이상, 또는 영문 2글자 이상
    tokens = re.findall(r"[\u4E00-\u9FFF\u30A0-\u30FF]{2,}|[a-zA-Z]{2,}", text)
    return [t for t in tokens if t.lower() not in sw]


def tokenize_text(text, lang, sw):
    """언어에 따른 적응적 토큰화"""
    if lang == "ko":
        return tokenize_korean(text, sw)
    elif lang == "en":
        return tokenize_english(text, sw)
    elif lang == "ja":
        return tokenize_japanese(text, sw)
    else:
        # 혼합/알 수 없음: 한국어 형태소 분석 시도
        return tokenize_korean(text, sw)

# ============================================================
# 3. 언어 감지
# ============================================================
from langdetect import detect

def detect_language(text):
    try:
        if not text or pd.isna(text) or len(str(text).strip()) < 5:
            return "unknown"
        return detect(str(text))
    except Exception:
        return "unknown"


# ============================================================
# 4. 3클래스 감성 분석
# ============================================================
def run_sentiment_3class(df, max_length=256):
    """3클래스 감성 분석: 긍정(>0.7) / 중립(0.3~0.7) / 부정(<0.3)"""
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("=" * 60)
    print("감성 분석 (KoELECTRA, 3클래스, max_length=%d)" % max_length)
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # MPS > CUDA > CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)
    model.eval()
    print(f"  device: {device}")

    pos_idx = 1  # 0=부정, 1=긍정
    texts = df["review_text"].fillna("").tolist()
    labels = []
    scores = []
    batch_size = 48

    for start in tqdm(range(0, len(texts), batch_size), desc="감성분석", unit="batch"):
        batch = [str(t)[:2000] for t in texts[start:start + batch_size]]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        for prob in probs:
            pos_score = float(prob[pos_idx])
            scores.append(pos_score)
            if pos_score >= 0.7:
                labels.append("긍정")
            elif pos_score <= 0.3:
                labels.append("부정")
            else:
                labels.append("중립")

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    print(f"\n  감성 분포:")
    vc = df["sentiment_label"].value_counts()
    for k, v in vc.items():
        print(f"    {k}: {v}건 ({v / len(df) * 100:.1f}%)")
    print()
    return df


# ============================================================
# 5. 감성 모델 검증
# ============================================================
def validate_sentiment(df):
    """별점 기준으로 감성 모델 정확도 검증"""
    from sklearn.metrics import confusion_matrix, classification_report

    def expected(r):
        if r >= 4:
            return "긍정"
        elif r == 3:
            return "중립"
        else:
            return "부정"

    valid = df.dropna(subset=["rating"]).copy()
    valid["expected"] = valid["rating"].apply(expected)

    labels_order = ["긍정", "중립", "부정"]
    cm = confusion_matrix(valid["expected"], valid["sentiment_label"], labels=labels_order)
    report = classification_report(
        valid["expected"], valid["sentiment_label"],
        labels=labels_order, zero_division=0, output_dict=True
    )
    report_str = classification_report(
        valid["expected"], valid["sentiment_label"],
        labels=labels_order, zero_division=0
    )

    match = (valid["expected"] == valid["sentiment_label"]).sum()
    accuracy = match / len(valid) * 100

    print("=" * 60)
    print("감성 모델 검증 (별점 기준)")
    print("=" * 60)
    print(f"  전체 정확도: {accuracy:.1f}%")
    print(f"\n{report_str}")

    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "report_str": report_str,
        "labels": labels_order,
    }


# ============================================================
# 6. LDA 토픽 모델링
# ============================================================
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def run_topic_modeling(texts, lang_list, sw, n_topics=5, n_words=8, label=""):
    """LDA 토픽 모델링"""
    print(f"  토픽 모델링 ({label}, {n_topics} topics)...")
    docs = []
    for text, lang in zip(texts, lang_list):
        tokens = tokenize_text(str(text), lang, sw)
        if tokens:
            docs.append(" ".join(tokens))
    if len(docs) < n_topics * 2:
        print(f"    문서 수 부족 ({len(docs)}건), 스킵")
        return None

    vectorizer = CountVectorizer(max_features=1000, min_df=2)
    dtm = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=42,
        max_iter=30, learning_method="online"
    )
    lda.fit(dtm)

    topics = []
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[::-1][:n_words]
        words = [feature_names[j] for j in top_idx]
        weights = [float(comp[j]) for j in top_idx]
        topics.append({"topic_id": i + 1, "words": words, "weights": weights})
        print(f"    Topic {i + 1}: {', '.join(words[:5])}")

    return topics


# ============================================================
# 7. 전체 분석 파이프라인
# ============================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[^0-9A-Za-z가-힣\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def analyze(name, csv_path, sw, img_prefix):
    """전체 분석 파이프라인 실행"""
    print("\n" + "=" * 70)
    print(f"  {name} 분석 시작")
    print("=" * 70 + "\n")

    # 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"데이터: {len(df)}건")

    # review_text 보장
    if "review_text" not in df.columns:
        df["review_text"] = df["content"].fillna("").astype(str)

    # 날짜 처리
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # rating 숫자 변환
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # ── 언어 감지 ──
    print("\n[1/6] 언어 감지...")
    df["detected_lang"] = df["review_text"].apply(detect_language)
    lang_counts = df["detected_lang"].value_counts().head(10)
    print(f"  언어 분포:\n{lang_counts.to_string()}\n")

    # ── 3클래스 감성 분석 ──
    print("[2/6] 감성 분석...")
    df = run_sentiment_3class(df, max_length=256)

    # ── 감성 모델 검증 ──
    print("[3/6] 모델 검증...")
    val = validate_sentiment(df)

    # ── 형태소 기반 키워드 분석 ──
    print("[4/6] 형태소 기반 키워드 분석...")
    pos_df = df[df["sentiment_label"] == "긍정"]
    neg_df = df[df["sentiment_label"] == "부정"]
    neu_df = df[df["sentiment_label"] == "중립"]

    pos_words, neg_words, neu_words = [], [], []
    for _, row in pos_df.iterrows():
        pos_words.extend(tokenize_text(row["review_text"], row.get("detected_lang", "ko"), sw))
    for _, row in neg_df.iterrows():
        neg_words.extend(tokenize_text(row["review_text"], row.get("detected_lang", "ko"), sw))
    for _, row in neu_df.iterrows():
        neu_words.extend(tokenize_text(row["review_text"], row.get("detected_lang", "ko"), sw))

    pos_counter = Counter(pos_words)
    neg_counter = Counter(neg_words)
    neu_counter = Counter(neu_words)

    print(f"  긍정 키워드 수: {len(pos_counter)}")
    print(f"  부정 키워드 수: {len(neg_counter)}")
    print(f"  중립 키워드 수: {len(neu_counter)}")
    print(f"  긍정 TOP10: {pos_counter.most_common(10)}")
    print(f"  부정 TOP10: {neg_counter.most_common(10)}")

    # ── LDA 토픽 모델링 ──
    print("\n[5/6] 토픽 모델링...")
    pos_topics = run_topic_modeling(
        pos_df["review_text"].tolist(), pos_df["detected_lang"].tolist(),
        sw, n_topics=5, label="긍정"
    )
    neg_topics = run_topic_modeling(
        neg_df["review_text"].tolist(), neg_df["detected_lang"].tolist(),
        sw, n_topics=5, label="부정"
    )

    # ── 시각화 ──
    print("\n[6/6] 시각화...")
    visualize_all(
        df, pos_counter, neg_counter, neu_counter,
        val, pos_topics, neg_topics,
        img_prefix, sw, name
    )

    # ── 언어별 감성 정확도 ──
    lang_accuracy = {}
    for lang in ["ko", "en", "ja"]:
        sub = df[df["detected_lang"] == lang]
        if len(sub) > 10:
            sub_valid = sub.dropna(subset=["rating"]).copy()
            expected = sub_valid["rating"].apply(lambda r: "긍정" if r >= 4 else ("부정" if r <= 2 else "중립"))
            match = (expected == sub_valid["sentiment_label"]).sum()
            lang_accuracy[lang] = {"count": len(sub), "accuracy": match / len(sub_valid) * 100 if len(sub_valid) > 0 else 0}

    print("\n  언어별 감성 정확도:")
    for lang, info in lang_accuracy.items():
        print(f"    {lang}: {info['count']}건, 정확도 {info['accuracy']:.1f}%")

    # ── CSV 저장 ──
    save_cols = ["platform", "country", "rating", "date", "review_text",
                 "detected_lang", "sentiment_label", "sentiment_score"]
    save_cols = [c for c in save_cols if c in df.columns]
    save_path = SENT_DIR / f"{img_prefix}_sentiment_v2.csv"
    df[save_cols].to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n  저장: {save_path}")

    return {
        "df": df,
        "validation": val,
        "pos_counter": pos_counter,
        "neg_counter": neg_counter,
        "neu_counter": neu_counter,
        "pos_topics": pos_topics,
        "neg_topics": neg_topics,
        "lang_accuracy": lang_accuracy,
    }


# ============================================================
# 8. 시각화
# ============================================================
def visualize_all(df, pos_counter, neg_counter, neu_counter, val,
                  pos_topics, neg_topics, prefix, sw, title):
    """모든 시각화 생성"""
    from wordcloud import WordCloud

    # 8-1. 별점 분포
    fig, ax = plt.subplots(figsize=(8, 5))
    rating_counts = df["rating"].value_counts().sort_index()
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
    bars = ax.bar(rating_counts.index, rating_counts.values, color=colors[:len(rating_counts)])
    for bar, val_cnt in zip(bars, rating_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val_cnt}\n({val_cnt / len(df) * 100:.1f}%)", ha="center", fontsize=10)
    ax.set_xlabel("별점")
    ax.set_ylabel("리뷰 수")
    ax.set_title(f"{title} 별점 분포 (n={len(df)}, 평균={df['rating'].mean():.2f})")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"{prefix}_rating_dist_v2.png", dpi=150)
    plt.close()

    # 8-2. 3클래스 감성 분포
    fig, ax = plt.subplots(figsize=(7, 5))
    sent_counts = df["sentiment_label"].value_counts()
    color_map = {"긍정": "#27ae60", "중립": "#f39c12", "부정": "#e74c3c"}
    labels_order = ["긍정", "중립", "부정"]
    vals = [sent_counts.get(l, 0) for l in labels_order]
    cols = [color_map[l] for l in labels_order]
    bars = ax.bar(labels_order, vals, color=cols)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v}건\n({v / len(df) * 100:.1f}%)", ha="center", fontsize=11)
    ax.set_title(f"{title} 감성 분포 (3클래스)")
    ax.set_ylabel("리뷰 수")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"{prefix}_sentiment_3class_v2.png", dpi=150)
    plt.close()

    # 8-3. 별점별 감성 비율 (stacked bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    ratings = sorted(df["rating"].dropna().unique())
    pos_pcts, neu_pcts, neg_pcts = [], [], []
    for r in ratings:
        sub = df[df["rating"] == r]
        total = len(sub)
        pos_pcts.append(len(sub[sub["sentiment_label"] == "긍정"]) / total * 100 if total else 0)
        neu_pcts.append(len(sub[sub["sentiment_label"] == "중립"]) / total * 100 if total else 0)
        neg_pcts.append(len(sub[sub["sentiment_label"] == "부정"]) / total * 100 if total else 0)
    x = np.arange(len(ratings))
    w = 0.6
    ax.bar(x, pos_pcts, w, label="긍정", color="#27ae60")
    ax.bar(x, neu_pcts, w, bottom=pos_pcts, label="중립", color="#f39c12")
    ax.bar(x, neg_pcts, w, bottom=[p + n for p, n in zip(pos_pcts, neu_pcts)], label="부정", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r)}점" for r in ratings])
    ax.set_ylabel("비율 (%)")
    ax.set_title(f"{title} 별점별 감성 비율 (3클래스)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"{prefix}_rating_sentiment_v2.png", dpi=150)
    plt.close()

    # 8-4. Confusion Matrix (모델 검증)
    fig, ax = plt.subplots(figsize=(7, 6))
    cm = val["confusion_matrix"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(val["labels"])
    ax.set_yticklabels(val["labels"])
    ax.set_xlabel("모델 예측")
    ax.set_ylabel("별점 기준 (기대값)")
    ax.set_title(f"{title} 감성 모델 Confusion Matrix\n(전체 정확도: {val['accuracy']:.1f}%)")
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"{prefix}_confusion_matrix_v2.png", dpi=150)
    plt.close()

    # 8-5. 키워드 TOP 30 (긍정/부정)
    for label, counter, color in [("긍정", pos_counter, "#27ae60"), ("부정", neg_counter, "#e74c3c")]:
        top30 = counter.most_common(30)
        if not top30:
            continue
        fig, ax = plt.subplots(figsize=(10, 8))
        words = [w for w, c in top30][::-1]
        counts = [c for w, c in top30][::-1]
        ax.barh(words, counts, color=color)
        ax.set_title(f"{title} {label} 키워드 TOP 30 (형태소 분석)")
        ax.set_xlabel("빈도")
        plt.tight_layout()
        tag = "pos" if label == "긍정" else "neg"
        plt.savefig(IMG_DIR / f"{prefix}_{tag}_top30_v2.png", dpi=150)
        plt.close()

    # 8-6. 워드클라우드 (긍정/부정)
    font_path = "/System/Library/Fonts/AppleGothic.ttf"
    for label, counter, cmap_name in [("긍정", pos_counter, "Greens"), ("부정", neg_counter, "Reds")]:
        if not counter:
            continue
        wc = WordCloud(
            font_path=font_path, width=1000, height=600,
            background_color="white", colormap=cmap_name,
            max_words=100, relative_scaling=0.3, min_font_size=10,
            stopwords=sw
        ).generate_from_frequencies(counter)
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{title} {label} 워드클라우드 (형태소 분석)", fontsize=14)
        plt.tight_layout()
        tag = "pos" if label == "긍정" else "neg"
        plt.savefig(IMG_DIR / f"{prefix}_wc_{tag}_v2.png", dpi=150)
        plt.close()

    # 8-7. 플랫폼별 비교
    if "platform" in df.columns and df["platform"].nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, plat in zip(axes, df["platform"].unique()):
            sub = df[df["platform"] == plat]
            sc = sub["sentiment_label"].value_counts()
            vals_plt = [sc.get(l, 0) for l in labels_order]
            cols_plt = [color_map[l] for l in labels_order]
            bars_plt = ax.bar(labels_order, vals_plt, color=cols_plt)
            for b, v in zip(bars_plt, vals_plt):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                        f"{v}건\n({v / len(sub) * 100:.1f}%)", ha="center", fontsize=10)
            ax.set_title(f"{plat} (n={len(sub)}, 평균={sub['rating'].mean():.2f})")
            ax.set_ylabel("리뷰 수")
        plt.suptitle(f"{title} 플랫폼별 감성 비교", fontsize=14)
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"{prefix}_platform_v2.png", dpi=150)
        plt.close()

    # 8-8. 국가별 감성 비교
    if "country" in df.columns and df["country"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        countries = df["country"].value_counts().index.tolist()
        x = np.arange(len(countries))
        w = 0.25
        for i, sent in enumerate(labels_order):
            vals_c = []
            for c in countries:
                sub = df[df["country"] == c]
                vals_c.append(len(sub[sub["sentiment_label"] == sent]) / len(sub) * 100 if len(sub) else 0)
            ax.bar(x + i * w, vals_c, w, label=sent, color=color_map[sent])
        ax.set_xticks(x + w)
        ax.set_xticklabels(countries)
        ax.set_ylabel("비율 (%)")
        ax.set_title(f"{title} 국가별 감성 비율 (3클래스)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(IMG_DIR / f"{prefix}_country_v2.png", dpi=150)
        plt.close()

    # 8-9. 언어별 감성 정확도
    if "detected_lang" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        lang_stats = []
        for lang in df["detected_lang"].value_counts().index[:6]:
            sub = df[df["detected_lang"] == lang].dropna(subset=["rating"])
            if len(sub) < 5:
                continue
            expected_map = sub["rating"].apply(lambda r: "긍정" if r >= 4 else ("부정" if r <= 2 else "중립"))
            acc = (expected_map == sub["sentiment_label"]).mean() * 100
            lang_stats.append((lang, len(sub), acc))
        if lang_stats:
            langs = [l[0] for l in lang_stats]
            counts_l = [l[1] for l in lang_stats]
            accs = [l[2] for l in lang_stats]
            bars_l = ax.bar(langs, accs, color="#3498db")
            for b, acc, cnt in zip(bars_l, accs, counts_l):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                        f"{acc:.1f}%\n(n={cnt})", ha="center", fontsize=10)
            ax.set_ylabel("정확도 (%)")
            ax.set_title(f"{title} 언어별 감성 모델 정확도")
            ax.set_ylim(0, 100)
            plt.tight_layout()
            plt.savefig(IMG_DIR / f"{prefix}_lang_accuracy_v2.png", dpi=150)
            plt.close()

    # 8-10. 토픽 모델링 시각화
    for label, topics, color in [("긍정", pos_topics, "#27ae60"), ("부정", neg_topics, "#e74c3c")]:
        if not topics:
            continue
        n = len(topics)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, topic in zip(axes, topics):
            words_t = topic["words"][::-1]
            w_arr = np.array(topic["weights"], dtype=float)
            weights_t = (w_arr / w_arr.max())[::-1]
            ax.barh(words_t, weights_t, color=color, alpha=0.8)
            ax.set_title(f"Topic {topic['topic_id']}", fontsize=11)
            ax.set_xlim(0, 1.1)
        plt.suptitle(f"{title} {label} 리뷰 토픽 (LDA)", fontsize=14)
        plt.tight_layout()
        tag = "pos" if label == "긍정" else "neg"
        plt.savefig(IMG_DIR / f"{prefix}_topics_{tag}_v2.png", dpi=150)
        plt.close()

    # 8-11. 월별 감성 추이
    if "date" in df.columns:
        df_dated = df.dropna(subset=["date"]).copy()
        df_dated["yearmonth"] = df_dated["date"].dt.to_period("M")
        if len(df_dated["yearmonth"].unique()) > 2:
            fig, ax = plt.subplots(figsize=(14, 6))
            monthly = df_dated.groupby("yearmonth")["sentiment_label"].value_counts().unstack(fill_value=0)
            for col in labels_order:
                if col not in monthly.columns:
                    monthly[col] = 0
            monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
            x_vals = range(len(monthly_pct))
            x_labels = [str(p) for p in monthly_pct.index]
            for sent in labels_order:
                ax.plot(x_vals, monthly_pct[sent].values, marker="o", label=sent,
                        color=color_map[sent], linewidth=2, markersize=4)
            ax.set_xticks(x_vals)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("비율 (%)")
            ax.set_title(f"{title} 월별 감성 추이 (3클래스)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(IMG_DIR / f"{prefix}_monthly_v2.png", dpi=150)
            plt.close()

    print(f"  시각화 완료: {IMG_DIR}/{prefix}_*.png")


# ============================================================
# 9. 비교 분석
# ============================================================
def compare_analysis(vrew_result, capcut_result):
    """Vrew vs CapCut 비교 분석"""
    print("\n" + "=" * 70)
    print("  Vrew vs CapCut 비교 분석")
    print("=" * 70)

    vdf = vrew_result["df"]
    cdf = capcut_result["df"]

    color_map = {"긍정": "#27ae60", "중립": "#f39c12", "부정": "#e74c3c"}
    labels_order = ["긍정", "중립", "부정"]

    # 9-1. 별점 분포 비교
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (df_c, name) in zip(axes, [(vdf, "Vrew"), (cdf, "CapCut")]):
        rc = df_c["rating"].value_counts().sort_index()
        colors = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"]
        bars = ax.bar(rc.index, rc.values, color=colors[:len(rc)])
        for b, v in zip(bars, rc.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{v / len(df_c) * 100:.1f}%", ha="center", fontsize=10)
        ax.set_title(f"{name} (n={len(df_c)}, 평균={df_c['rating'].mean():.2f})")
        ax.set_xlabel("별점")
        ax.set_ylabel("리뷰 수")
    plt.suptitle("Vrew vs CapCut 별점 분포 비교", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "vs_rating_dist_v2.png", dpi=150)
    plt.close()

    # 9-2. 3클래스 감성 비교
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    w = 0.35
    vsc = vdf["sentiment_label"].value_counts()
    csc = cdf["sentiment_label"].value_counts()
    v_vals = [vsc.get(l, 0) / len(vdf) * 100 for l in labels_order]
    c_vals = [csc.get(l, 0) / len(cdf) * 100 for l in labels_order]
    ax.bar(x - w / 2, v_vals, w, label="Vrew", color="#3498db")
    ax.bar(x + w / 2, c_vals, w, label="CapCut", color="#e74c3c")
    for i, (v, c) in enumerate(zip(v_vals, c_vals)):
        ax.text(i - w / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
        ax.text(i + w / 2, c + 0.5, f"{c:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_order)
    ax.set_ylabel("비율 (%)")
    ax.set_title("Vrew vs CapCut 감성 비교 (3클래스)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / "vs_sentiment_3class_v2.png", dpi=150)
    plt.close()

    # 9-3. 부정 키워드 비교
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (counter, name) in zip(axes, [
        (vrew_result["neg_counter"], "Vrew"),
        (capcut_result["neg_counter"], "CapCut")
    ]):
        top20 = counter.most_common(20)
        if top20:
            words = [w for w, c in top20][::-1]
            counts = [c for w, c in top20][::-1]
            ax.barh(words, counts, color="#e74c3c")
        ax.set_title(f"{name} 부정 키워드 TOP 20")
        ax.set_xlabel("빈도")
    plt.suptitle("Vrew vs CapCut 부정 키워드 비교 (형태소 분석)", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "vs_neg_keywords_v2.png", dpi=150)
    plt.close()

    # 9-4. 긍정 키워드 비교
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, (counter, name) in zip(axes, [
        (vrew_result["pos_counter"], "Vrew"),
        (capcut_result["pos_counter"], "CapCut")
    ]):
        top20 = counter.most_common(20)
        if top20:
            words = [w for w, c in top20][::-1]
            counts = [c for w, c in top20][::-1]
            ax.barh(words, counts, color="#27ae60")
        ax.set_title(f"{name} 긍정 키워드 TOP 20")
        ax.set_xlabel("빈도")
    plt.suptitle("Vrew vs CapCut 긍정 키워드 비교 (형태소 분석)", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "vs_pos_keywords_v2.png", dpi=150)
    plt.close()

    # 9-5. 국가별 감성 비교
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (df_c, name) in zip(axes, [(vdf, "Vrew"), (cdf, "CapCut")]):
        countries = df_c["country"].value_counts().index.tolist()
        x_c = np.arange(len(countries))
        w_c = 0.25
        for i, sent in enumerate(labels_order):
            vals_cc = []
            for c in countries:
                sub = df_c[df_c["country"] == c]
                vals_cc.append(len(sub[sub["sentiment_label"] == sent]) / len(sub) * 100 if len(sub) else 0)
            ax.bar(x_c + i * w_c, vals_cc, w_c, label=sent, color=color_map[sent])
        ax.set_xticks(x_c + w_c)
        ax.set_xticklabels(countries)
        ax.set_ylabel("비율 (%)")
        ax.set_title(f"{name}")
        ax.legend(fontsize=8)
    plt.suptitle("Vrew vs CapCut 국가별 감성 비교 (3클래스)", fontsize=14)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "vs_country_v2.png", dpi=150)
    plt.close()

    # 9-6. 모델 정확도 비교
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ["Vrew", "CapCut"]
    accs = [vrew_result["validation"]["accuracy"], capcut_result["validation"]["accuracy"]]
    bars = ax.bar(names, accs, color=["#3498db", "#e74c3c"])
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{a:.1f}%", ha="center", fontsize=12)
    ax.set_ylabel("정확도 (%)")
    ax.set_title("감성 모델 정확도 비교 (별점 기준)")
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(IMG_DIR / "vs_model_accuracy_v2.png", dpi=150)
    plt.close()

    print("  비교 시각화 완료")

    return {
        "vrew_sentiment": vdf["sentiment_label"].value_counts().to_dict(),
        "capcut_sentiment": cdf["sentiment_label"].value_counts().to_dict(),
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    vrew_result = None
    capcut_result = None

    if mode in ("vrew", "all"):
        vrew_result = analyze(
            "Vrew",
            CSV_DIR / "vrew_reviews_merged_2026.csv",
            stopwords_vrew,
            "vrew"
        )

    if mode in ("capcut", "all"):
        capcut_result = analyze(
            "CapCut",
            CSV_DIR / "capcut_reviews_2026.csv",
            stopwords_capcut,
            "capcut"
        )

    if mode in ("compare", "all"):
        if vrew_result is None:
            print("Vrew 결과 로드...")
            vrew_result = analyze("Vrew", CSV_DIR / "vrew_reviews_merged_2026.csv", stopwords_vrew, "vrew")
        if capcut_result is None:
            print("CapCut 결과 로드...")
            capcut_result = analyze("CapCut", CSV_DIR / "capcut_reviews_2026.csv", stopwords_capcut, "capcut")
        compare_analysis(vrew_result, capcut_result)

    print("\n" + "=" * 70)
    print("  모든 분석 완료!")
    print("=" * 70)
