import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

# ===== 1. 설정 =====
CSV_PATH = "/Users/seojeong-il/Desktop/내문서/데이터 분석/개인 분석/보이저엑스/vrew/sentiment_out/reviews_with_sentiment.csv"  # 네 CSV 파일 이름/경로
TEXT_COL = "review_text"                 # 리뷰 텍스트 컬럼명
SENT_COL = "Sentiment_label"             # 감정 라벨 컬럼명

# ✅ 네 데이터 기준: '긍정' / '부정'
POS_VALUE = "긍정"
NEG_VALUE = "부정"

# Mac 기본 한글 폰트
FONT_PATH = "/System/Library/Fonts/AppleGothic.ttf"

# ===== 2. 데이터 로드 =====
df = pd.read_csv(CSV_PATH)

# 라벨 값 확인용 (한번 실행해보고 괜찮으면 주석 처리해도 됨)
print("컬럼:", df.columns.tolist())
print("라벨 분포:")
print(df[SENT_COL].value_counts())

# ===== 3. 간단한 전처리 함수 =====
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # 이모지/특수문자 제거 (한글, 영어, 숫자, 공백만 남김)
    text = re.sub(r"[^0-9A-Za-z가-힣 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

# ===== 4. 긍/부정 텍스트 합치기 =====
pos_text = " ".join(
    df.loc[df[SENT_COL] == POS_VALUE, TEXT_COL].dropna().tolist()
)
neg_text = " ".join(
    df.loc[df[SENT_COL] == NEG_VALUE, TEXT_COL].dropna().tolist()
)

print(f"긍정 리뷰 텍스트 길이: {len(pos_text)}")
print(f"부정 리뷰 텍스트 길이: {len(neg_text)}")

# ===== 5. 워드클라우드 생성 함수 =====
def make_wordcloud(text, output_file, title):
    if not text.strip():
        print(f"[경고] {title} 텍스트가 비어 있어서 워드클라우드를 만들 수 없습니다.")
        return

    wc = WordCloud(
        font_path=FONT_PATH,
        width=1600,
        height=800,
        background_color="white",
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()

    wc.to_file(output_file)
    print(f"✅ {title} 워드클라우드 저장 완료 → {output_file}")

# ===== 6. 실제 생성 =====
make_wordcloud(pos_text, "wordcloud_positive.png", "긍정 리뷰 워드클라우드")
make_wordcloud(neg_text, "wordcloud_negative.png", "부정 리뷰 워드클라우드")

