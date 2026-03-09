# Vrew 리뷰 데이터 분석

Vrew(브루) 서비스의 App Store / Google Play 사용자 리뷰를 수집하고, 감성 분석을 통해 주요 불만 요소와 만족 포인트를 파악한 프로젝트입니다.

## 분석 결과 요약

| 항목 | 수치 |
|------|------|
| 총 리뷰 수 | 570건 |
| 분석 기간 | 2019.10 ~ 2026.03 |
| 수집 플랫폼 | App Store, Google Play |
| 수집 국가 | 한국(kr), 일본(jp), 미국(us) |
| 평균 별점 | 3.89점 |
| 긍정률 | 76.8% |

### 주요 인사이트

- **핵심 강점**: 자동 자막 생성 기능에 대한 높은 만족도
- **핵심 불만**: AI/음성 인식 품질, 앱 안정성(크래시), fps 문제, 내보내기 기능
- **플랫폼 차이**: Google Play(부정률 29.5%)가 App Store(16.7%) 대비 불만 2배
- **국가별 차이**: 한국(긍정률 67.3%)이 일본(86.6%), 미국(96.8%) 대비 가장 낮음

> 자세한 내용은 [분석결과_2026.md](분석결과_2026.md), [기존_신규_비교분석.md](기존_신규_비교분석.md) 참고

## 시각화 미리보기

### 별점 분포
![별점 분포](분석%20이미지%20파일/rating_distribution_2026.png)

### 월별 감성 추이
![월별 감성 추이](분석%20이미지%20파일/monthly_sentiment_2026.png)

### 플랫폼별 별점 비교
![플랫폼별 비교](분석%20이미지%20파일/platform_comparison_2026.png)

### 국가별 감성 분포
![국가별 감성](분석%20이미지%20파일/country_sentiment_2026.png)

### 긍정/부정 키워드 TOP 30
| 긍정 | 부정 |
|------|------|
| ![긍정 TOP30](분석%20이미지%20파일/pos_top30_2026.png) | ![부정 TOP30](분석%20이미지%20파일/neg_top30_2026.png) |

### 워드클라우드
| 긍정 | 부정 |
|------|------|
| ![긍정 워드클라우드](분석%20이미지%20파일/wordcloud_positive_2026.png) | ![부정 워드클라우드](분석%20이미지%20파일/wordcloud_negative_2026.png) |

## 프로젝트 구조

```
vrew/
├── README.md                     # 프로젝트 소개
├── 분석결과_2026.md               # 전체 분석 결과 리포트
├── 기존_신규_비교분석.md           # 기존(2025.11) vs 신규(2026.03) 비교
├── CSV 데이터/
│   └── vrew_reviews_merged_2026.csv   # 최종 병합 데이터 (570건)
├── sentiment_out/
│   └── reviews_with_sentiment_2026.csv  # 감성 분석 결과
├── 분석 이미지 파일/
│   ├── rating_distribution_2026.png     # 별점 분포
│   ├── pos_top30_2026.png              # 긍정 키워드 TOP30
│   ├── neg_top30_2026.png              # 부정 키워드 TOP30
│   ├── wordcloud_positive_2026.png     # 긍정 워드클라우드
│   ├── wordcloud_negative_2026.png     # 부정 워드클라우드
│   ├── monthly_sentiment_2026.png      # 월별 감성 추이
│   ├── platform_comparison_2026.png    # 플랫폼별 비교
│   ├── country_sentiment_2026.png      # 국가별 감성 분포
│   ├── comparison_rating_distribution.png  # 기존vs신규 별점 비교
│   ├── comparison_sentiment.png            # 기존vs신규 감성 비교
│   └── comparison_positivity_by_rating.png # 기존vs신규 긍정률 비교
└── 분석/크롤링/테스트 코드/
    ├── 크롤링_2026.py             # 리뷰 크롤링 스크립트
    ├── 통합_분석_2026.py           # 전처리 + 감성분석 + 시각화 통합
    └── 브류 리뷰 크롤링.py         # 원본 크롤링 코드 (참고용)
```

## 분석 파이프라인

1. **데이터 수집** — App Store RSS + Google Play Scraper (kr/us/jp)
2. **전처리** — 병합, 중복 제거, 텍스트 정제
3. **감성 분석** — KoELECTRA 모델 (jaehyeong/koelectra-base-v3-generalized-sentiment-analysis)
4. **시각화** — 별점 분포, 키워드 빈도, 워드클라우드, 시계열 추이, 플랫폼/국가별 비교

## 실행 방법

```bash
# 패키지 설치
pip install requests google-play-scraper pandas torch transformers wordcloud matplotlib tqdm

# 1. 크롤링
python 분석/크롤링/테스트\ 코드/크롤링_2026.py

# 2. 분석 (전처리 + 감성분석 + 시각화)
python 분석/크롤링/테스트\ 코드/통합_분석_2026.py
```

## 기술 스택

- Python 3.9+
- pandas, matplotlib, wordcloud
- google-play-scraper
- HuggingFace Transformers (KoELECTRA)
