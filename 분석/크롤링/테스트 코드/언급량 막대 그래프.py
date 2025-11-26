import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import sys

# ===== 1. 설정 =====
CSV_PATH = "/Users/seojeong-il/Desktop/내문서/데이터 분석/개인 분석/보이저엑스/vrew/sentiment_out/reviews_with_sentiment.csv"
TEXT_COL = "review_text"
SENT_COL = "Sentiment_label"

POS_VALUE = "긍정"
NEG_VALUE = "부정"

# 한글 폰트 설정
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

# ===== 2. 데이터 로드 (에러 처리) =====
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✓ 데이터 로드 완료: {len(df)}개 행")
    
    # 필수 컬럼 확인
    if TEXT_COL not in df.columns or SENT_COL not in df.columns:
        print(f"❌ 필수 컬럼이 없습니다: {TEXT_COL}, {SENT_COL}")
        sys.exit(1)
        
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {CSV_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 데이터 로드 오류: {e}")
    sys.exit(1)

# ===== 3. 텍스트 정제 =====
def clean_text(text):
    """특수문자 제거 및 공백 정리"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[^0-9A-Za-z가-힣 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df[TEXT_COL] = df[TEXT_COL].apply(clean_text)

# ===== 4. 불용어 정의 =====
# 영어 불용어
english_stopwords = set("""
the to of and in is it that for on with as are be was were at by this from 
or but about not into up out over after so than then too can an no all would 
there their what when which who how has had have will your more if my me do
""".split())

# 한글 불용어 (일반 + 감정 표현)
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

# 통합
stopwords = english_stopwords | korean_stopwords | domain_stopwords

# ===== 5. 토큰 추출 =====
def tokenize(text):
    """단어 추출 (한글 2자 이상, 영어 2자 이상)"""
    text = clean_text(text)
    # 한글 단어 (2자 이상) + 영어 단어 (2자 이상)
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", text)
    # 불용어 제거 (영어는 소문자 변환 후 비교)
    tokens = [t for t in tokens if t.lower() not in stopwords]
    return tokens

# ===== 6. 긍정/부정 분리 =====
pos_df = df[df[SENT_COL] == POS_VALUE]
neg_df = df[df[SENT_COL] == NEG_VALUE]

pos_texts = pos_df[TEXT_COL].tolist()
neg_texts = neg_df[TEXT_COL].tolist()

print(f"\n✓ 긍정 리뷰: {len(pos_texts)}개")
print(f"✓ 부정 리뷰: {len(neg_texts)}개")

# ===== 7. 단어 카운트 =====
pos_words = []
neg_words = []

for t in pos_texts:
    pos_words.extend(tokenize(t))

for t in neg_texts:
    neg_words.extend(tokenize(t))

print(f"\n✓ 긍정 단어 수: {len(pos_words):,}개 (유니크: {len(set(pos_words)):,}개)")
print(f"✓ 부정 단어 수: {len(neg_words):,}개 (유니크: {len(set(neg_words)):,}개)")

pos_freq = Counter(pos_words).most_common(30)
neg_freq = Counter(neg_words).most_common(30)

# 상위 5개 출력
print("\n긍정 TOP 5:", pos_freq[:5])
print("부정 TOP 5:", neg_freq[:5])

# ===== 8. 시각화 함수 =====
def plot_top(freq_data, title, output_file, color):
    """단어 빈도 막대 그래프"""
    if not freq_data:
        print(f"⚠️ 표시할 데이터가 없습니다: {title}")
        return
    
    words = [w for w, c in freq_data]
    counts = [c for w, c in freq_data]

    plt.figure(figsize=(10, 10))
    plt.barh(words[::-1], counts[::-1], color=color, edgecolor='white', linewidth=0.7)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("빈도수", fontsize=12)
    plt.ylabel("단어", fontsize=12)
    
    # 그리드 추가
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n✓ 저장 완료 → {output_file}")

# ===== 9. 그래프 생성 =====
plot_top(pos_freq, "긍정 리뷰 단어 TOP 30", "pos_top30_cleaned.png", "#4CAF50")
plot_top(neg_freq, "부정 리뷰 단어 TOP 30", "neg_top30_cleaned.png", "#F44336")

print("\n✓ 모든 작업 완료!")


