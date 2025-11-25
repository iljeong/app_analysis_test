import os
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path("/Users/seojeong-il/Desktop/내문서/데이터 분석/개인 분석/보이저엑스/vrew")
MPLCONFIG_DIR = BASE_DIR / ".mplconfig"
CACHE_DIR = BASE_DIR / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
except ImportError as exc:
    raise SystemExit("matplotlib이 설치되지 않아 시각화를 실행할 수 없습니다.") from exc

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

CSV_PATH = str(BASE_DIR / "vrew_reviews_combined.csv")
CLEAN_PATH = str(BASE_DIR / "vrew_reviews_clean.csv")
TOKEN_CSV_PATH = str(BASE_DIR / "vrew_reviews_tokens.csv")
PLOT_PATH = str(BASE_DIR / "rating_distribution.png")

STRING_COLS = [
    "platform",
    "author",
    "title",
    "content",
    "version",
    "country",
    "reviewId",
    "userImage",
    "reviewCreatedVersion",
    "replyContent",
    "repliedAt",
    "appVersion",
    "lang",
]

NUMERIC_COLS = [
    "rating",
    "vote_sum",
    "vote_count",
    "review_id",
    "thumbsUpCount",
]

DATE_COLS = ["updated", "at"]

EXCLUDE_KEYWORDS = ["일레클", "딜카", "패스카", "카카오", "모빌리티"]
EXCLUDE_PATTERN = re.compile(
    "(?:" + "|".join(map(re.escape, EXCLUDE_KEYWORDS)) + r")(?:은|는|이|가|을|를|에|에서|으로|와|과|의|도)?",
    flags=re.IGNORECASE,
)

STOPWORDS = set([
    "하다","되다","이다","있다","없다","같다","보다","주다","받다","되",
    "좋다","나쁘다","자다","됨","되고","해서","하면","하는","했다","했던",
    "같은","든","다시","예","아이고","아휴","하","허","후",
    "의","가","이","은","는","을","를","에","로","도","만","와","과","및",
    "그리고","마다","에서","으로","에게","까지","때문","때문에","지만","거나",
    "때","거","수","정말","너무","매우","진짜","완전","계속","그냥","좀","잘",
    "더","덜","막","또","등","라","데","요","니다","그린","이용","사용","서비스",
    "차량","운전","오늘","어제","내일","이번","저번","다른","없고","없다고",
    "없음","없이","없는","근데","그럼","전에","그렇게","이게","이런","이렇게",
    "처음","바로","지금","결국","무슨","절대","많이","전혀","아니","아니고","아예",
    "안되고","안되서","안된다고","안됨","안받고","못하고","해도","했더니","대한",
    "저는","제가","내가","회사","고객이","합니다","문이","차가","차를","차량이",
    "차량을","그린카","그린카는","쏘카","쏘카는","쏘카를","사용하다","그렇게",
    "갑자기","있어서","이용하다","없어서","거예요","거네요","거같아요","거같음",
    "아","어","음","헐","휴","우와","에휴","진짜로","와","흠","아니요","네","응",
    "그래서","그러니까","그런데","그럼에도","그러면","아니면","때문인지",
    "때문인지도","왜냐면","그렇지만","또한","그리고나서","그래도","그런가",
    "그랬더니","그렇다보니","하게","하면서도","하려고","하려니",
    "까지는","처럼","정도","대로","뿐","따라","마다","부터","만큼","하면서",
    "그나마","조금","살짝","약간","되게","엄청","너무나","굉장히","항상",
    "계속해서","매번","대체로","거의","아주","완전히","도대체",
    "좀더","빨리","늦게","처리","제발","문의","요청","답변","불편",
    "문의했는데","문의드려요","개선","필요","해결","조치","문제","상황",
    "도요","으로는","에는","에게는","라고","이라고","네요","습니다","는데",
    "는데요","해서요","같아요","해요","하네요","네요.","같음","거에요"
])


def get_tokenizer():
    try:
        from konlpy.tag import Okt  # type: ignore
        okt = Okt()

        def tokenize_ko(text: str):
            text = str(text)
            morphs = []
            for w, pos in okt.pos(text, norm=True, stem=True):
                if pos == "Noun" and len(w) > 1:
                    morphs.append(w)
            return morphs

        print("[INFO] 토크나이저: Okt 사용")
        return tokenize_ko

    except Exception as exc:
        print(f"[WARN] Okt 사용 불가, fallback 토크나이저 사용: {exc}")

        def simple_tokenize_ko(text: str):
            text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", str(text))
            return re.findall(r"[가-힣]{2,}", text)

        return simple_tokenize_ko


def clean_text(text: str) -> str:
    text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_and_filter(tokenizer, text: str):
    toks = tokenizer(text)
    return [w for w in toks if w not in STOPWORDS and len(w) > 1]

def main():
    df = pd.read_csv(CSV_PATH)
    df["review_text"] = df.get("content", "")

    print("=== NaN 개수 (원본) ===")
    print(df.isna().sum())
    print()

    df_clean = df.copy()

    for col in STRING_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("")

    for col in NUMERIC_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)

    for col in DATE_COLS:
        if col in df_clean.columns:
            dt_series = pd.to_datetime(df_clean[col], errors="coerce")
            df_clean[col] = dt_series.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

    print("=== NaN 개수 (전처리 후) ===")
    print(df_clean.replace("", pd.NA).isna().sum())
    print()

    df_clean.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")
    print(f"전처리 완료 → {CLEAN_PATH}")

    tokenizer = get_tokenizer()

    before = len(df_clean)
    df_filtered = df_clean[~df_clean["review_text"].astype(str).str.contains(EXCLUDE_PATTERN, na=False)].copy()
    df_filtered.reset_index(drop=True, inplace=True)
    print(f"[INFO] 타 서비스 언급 제거: {before - len(df_filtered)}건 제거, 잔여 {len(df_filtered):,}건")

    df_filtered["clean_text"] = df_filtered["review_text"].apply(clean_text)
    df_filtered["tokens"] = df_filtered["clean_text"].apply(lambda text: tokenize_and_filter(tokenizer, text))
    df_filtered["tokens_str"] = df_filtered["tokens"].apply(lambda xs: " ".join(xs))

    if "updated" in df_filtered.columns:
        df_filtered["updated"] = pd.to_datetime(df_filtered["updated"], errors="coerce").dt.date.astype(str)
    if "at" in df_filtered.columns:
        df_filtered["at"] = pd.to_datetime(df_filtered["at"], errors="coerce").dt.date.astype(str)

    df_filtered.to_csv(TOKEN_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] 토큰/불용어 전처리 결과 저장 → {TOKEN_CSV_PATH}")

    if "rating" in df_clean.columns:
        plt.figure(figsize=(8, 4))
        rating_order = sorted(df_clean["rating"].unique())
        rating_counts = df_clean["rating"].value_counts().reindex(rating_order, fill_value=0)
        rating_counts.plot(kind="bar", color="#5B8FF9")
        plt.title("별점 분포")
        plt.xlabel("별점")
        plt.ylabel("리뷰 수")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(PLOT_PATH, dpi=150)
        print(f"별점 분포 그래프 저장 → {PLOT_PATH}")
        plt.close()


if __name__ == "__main__":
    main()