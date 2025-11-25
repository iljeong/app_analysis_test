import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# ===== 1. 설정 =====
CSV_PATH = "/Users/seojeong-il/Desktop/내문서/데이터 분석/개인 분석/보이저엑스/vrew/sentiment_out/reviews_with_sentiment.csv"
TEXT_COL = "review_text"
SENT_COL = "Sentiment_label"

POS_VALUE = "긍정"
NEG_VALUE = "부정"

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ===== 2. 데이터 로드 =====
df = pd.read_csv(CSV_PATH)

# ===== 3. 텍스트 정제 =====
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[^0-9A-Za-z가-힣 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

# ===== 4. 불용어 수정(영어 + 한글 + 도메인) =====

# 영어 불용어 (가장 일반적인 100개 정도)
english_stopwords = set("""
the to of and in is it that for on with as are be was were at by this from or but about not into up out over after into so than then too can an
""".split())

# 한글 불용어
korean_stopwords = set("""
그리고 하지만 그러나 이런 저런 그냥 너무 정말 진짜 거의 근데 그래서 사용 이용 때문 이건 저건 그건 때 것 거 좀 더 등 듯 거요 거에요
이번 다음 현재 오늘 어제 저희 서비스 고객
좋아요 감사합니다 있습니다 좋습니다 합니다 해요 되는 있어요
""".split())

# 도메인 불용어 (원하면 조절)  
domain_stopwords = set("""
자막 영상 pc 동영상 android ios 앱 기능 모바일 화면 버전 update 업데이트
""".split())

stopwords = english_stopwords | korean_stopwords | domain_stopwords

# ===== 5. 토큰 추출 =====
def tokenize(text):
    text = clean_text(text)
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", text)
    tokens = [t for t in tokens if t.lower() not in stopwords]
    return tokens

# ===== 6. 긍정/부정 분리 =====
pos_texts = df.loc[df[SENT_COL] == POS_VALUE, TEXT_COL].tolist()
neg_texts = df.loc[df[SENT_COL] == NEG_VALUE, TEXT_COL].tolist()

# ===== 7. 단어 카운트 =====
pos_words = []
neg_words = []

for t in pos_texts:
    pos_words.extend(tokenize(t))

for t in neg_texts:
    neg_words.extend(tokenize(t))

pos_freq = Counter(pos_words).most_common(30)
neg_freq = Counter(neg_words).most_common(30)

# ===== 8. 그래프 함수 =====
def plot_top(freq_data, title, output_file, color):
    words = [w for w, c in freq_data]
    counts = [c for w, c in freq_data]

    plt.figure(figsize=(10, 10))
    plt.barh(words[::-1], counts[::-1], color=color)
    plt.title(title)
    plt.xlabel("빈도수")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.show()

    print(f"저장 완료 → {output_file}")

# ===== 9. 저장 =====
plot_top(pos_freq, "긍정 리뷰 단어 TOP 30 (정제 후)", "pos_top30_cleaned.png", "#4CAF50")
plot_top(neg_freq, "부정 리뷰 단어 TOP 30 (정제 후)", "neg_top30_cleaned.png", "#F44336")


