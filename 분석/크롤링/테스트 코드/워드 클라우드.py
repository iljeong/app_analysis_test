import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import sys

# ===== 1. 설정 =====
CSV_PATH = "/Users/seojeong-il/Desktop/내문서/데이터 분석/개인 분석/보이저엑스/vrew/sentiment_out/reviews_with_sentiment.csv"
TEXT_COL = "review_text"
SENT_COL = "Sentiment_label"

POS_VALUE = "긍정"
NEG_VALUE = "부정"

# Mac 기본 한글 폰트
FONT_PATH = "/System/Library/Fonts/AppleGothic.ttf"

# ===== 2. 불용어 정의 (빈도 분석과 동일) =====
# 영어 불용어
english_stopwords = set("""
the to of and in is it that for on with as are be was were at by this from 
or but about not into up out over after so than then too can an no all would 
there their what when which who how has had have will your more if my me do
""".split())

# 한글 불용어
korean_stopwords = set("""
그리고 하지만 그러나 그런 이런 저런 그냥 너무 정말 진짜 거의 
근데 그래서 때문 이건 저건 그건 때 것 거 좀 더 등 듯 
이번 다음 현재 오늘 어제 저희 우리 제가 
있습니다 좋습니다 합니다 됩니다 해요 되는 있어요 되요 
좋아요 감사합니다 감사해요 대단히 정말로
같은 같이 처럼 보다 만큼 이나 라도 라서 니까
""".split())

# 도메인 특화 불용어 (필요시 조정)
domain_stopwords = set("""
자막 영상 동영상 앱 기능 화면 버전 업데이트 update
""".split())

# 통합 불용어
stopwords = english_stopwords | korean_stopwords | domain_stopwords

# ===== 3. 데이터 로드 (에러 처리) =====
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✓ 데이터 로드 완료: {len(df)}개 행\n")
    
    # 필수 컬럼 확인
    if TEXT_COL not in df.columns or SENT_COL not in df.columns:
        print(f"❌ 필수 컬럼이 없습니다: {TEXT_COL}, {SENT_COL}")
        sys.exit(1)
    
    # 라벨 분포 확인
    print("감정 라벨 분포:")
    print(df[SENT_COL].value_counts())
    print()
        
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {CSV_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 데이터 로드 오류: {e}")
    sys.exit(1)

# ===== 4. 텍스트 전처리 =====
def clean_text(text):
    """특수문자 제거 및 공백 정리"""
    if pd.isna(text):
        return ""
    text = str(text)
    # 한글, 영어, 숫자, 공백만 남김
    text = re.sub(r"[^0-9A-Za-z가-힣 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

# ===== 5. 불용어 필터링 함수 =====
def filter_stopwords(text):
    """텍스트에서 불용어 제거"""
    words = text.split()
    # 2자 이상 단어만 유지 & 불용어 제거
    filtered = [
        word for word in words 
        if len(word) >= 2 and word.lower() not in stopwords
    ]
    return " ".join(filtered)

# ===== 6. 긍/부정 텍스트 합치기 및 불용어 제거 =====
pos_text_raw = " ".join(
    df.loc[df[SENT_COL] == POS_VALUE, TEXT_COL].dropna().tolist()
)
neg_text_raw = " ".join(
    df.loc[df[SENT_COL] == NEG_VALUE, TEXT_COL].dropna().tolist()
)

# 불용어 필터링 적용
pos_text = filter_stopwords(pos_text_raw)
neg_text = filter_stopwords(neg_text_raw)

print(f"긍정 리뷰 텍스트 길이: {len(pos_text):,}자 (필터링 전: {len(pos_text_raw):,}자)")
print(f"부정 리뷰 텍스트 길이: {len(neg_text):,}자 (필터링 전: {len(neg_text_raw):,}자)")
print()

# ===== 7. 워드클라우드 생성 함수 =====
def make_wordcloud(text, output_file, title, colormap):
    """워드클라우드 생성 및 저장"""
    if not text.strip():
        print(f"⚠️ {title} 텍스트가 비어 있어서 워드클라우드를 만들 수 없습니다.")
        return

    wc = WordCloud(
        font_path=FONT_PATH,
        width=1600,
        height=800,
        background_color="white",
        colormap=colormap,           # 색상 테마 추가
        max_words=200,               # 최대 단어 수
        relative_scaling=0.3,        # 빈도 차이 시각화 강도
        min_font_size=10,            # 최소 글자 크기
        stopwords=stopwords          # 추가 보험용 (중복 제거)
    ).generate(text)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    wc.to_file(output_file)
    print(f"✓ 저장 완료 → {output_file}\n")

# ===== 8. 워드클라우드 생성 =====
make_wordcloud(
    pos_text, 
    "wordcloud_positive.png", 
    "긍정 리뷰 워드클라우드 (불용어 제거)",
    "Greens"  # 초록 계열
)

make_wordcloud(
    neg_text, 
    "wordcloud_negative.png", 
    "부정 리뷰 워드클라우드 (불용어 제거)",
    "Reds"  # 빨강 계열
)

print("✓ 모든 작업 완료!")

