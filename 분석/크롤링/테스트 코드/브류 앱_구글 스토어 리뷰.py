# 필요한 패키지가 없다면 터미널에서 아래 명령을 수동으로 실행하세요.
# pip install requests google-play-scraper pandas

# -*- coding: utf-8 -*-
"""
앱스토어(Apple) + 구글플레이 리뷰 통합 크롤러 (Vrew 예시)
- 입력: 각 마켓의 앱 URL
- 처리:
    1) URL에서 앱 ID 자동 추출
    2) 앱스토어 RSS 리뷰 수집
    3) 구글플레이 reviews_all로 리뷰 수집
    4) 각각 CSV 저장 + 통합 CSV 저장
"""

import requests
import pandas as pd
import time
from urllib.parse import urlparse, parse_qs
from google_play_scraper import reviews_all


# ===============================
# 1. URL에서 ID 추출 함수
# ===============================

def get_appstore_id_from_url(url: str) -> str:
    """
    App Store URL에서 숫자 앱 ID 추출
    예) https://apps.apple.com/kr/app/vrew-브루/id1477811799
        -> 1477811799
    """
    path_parts = urlparse(url).path.split("/")
    # 뒤에서부터 'id123456789' 형태 찾기
    for part in reversed(path_parts):
        if part.startswith("id") and part[2:].isdigit():
            return part[2:]
    # 혹시 모를 대비: 숫자만 있는 부분 찾기
    for part in reversed(path_parts):
        if part.isdigit():
            return part
    raise ValueError("App Store ID를 URL에서 찾을 수 없습니다.")


def get_gplay_id_from_url(url: str) -> str:
    """
    Google Play URL에서 패키지명(app id) 추출
    예) https://play.google.com/store/apps/details?id=com.voyagerx.vrew.android
        -> com.voyagerx.vrew.android
    """
    qs = parse_qs(urlparse(url).query)
    if "id" in qs:
        return qs["id"][0]
    raise ValueError("Google Play app id를 URL에서 찾을 수 없습니다.")


# ===============================
# 2. 앱스토어 리뷰 수집 함수
# ===============================

def fetch_app_store_reviews(app_id: str,
                            country: str = "kr",
                            pages: int = 10,
                            sleep_sec: float = 1.0) -> pd.DataFrame:
    """
    Apple App Store RSS를 이용해 리뷰 가져오기
    - app_id: 숫자 ID (예: '1477811799')
    - country: 스토어 국가 코드 (kr, us 등)
    - pages: 가져올 최대 페이지 수 (page=1~pages)
    """
    all_reviews = []

    for page in range(1, pages + 1):
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"page={page}/id={app_id}/sortby=mostrecent/json"
        )
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"[AppStore] page {page} 요청 실패(status={resp.status_code}), 종료")
            break

        data = resp.json()
        # 리뷰가 없거나 마지막 페이지면 종료
        if "feed" not in data or "entry" not in data["feed"]:
            print(f"[AppStore] page {page}에 더 이상 리뷰 없음, 종료")
            break

        entries = data["feed"]["entry"]

        # 첫 entry는 앱 메타정보인 경우가 많아서 rating 없는 것 제외
        reviews_this_page = 0
        for e in entries:
            if "im:rating" not in e:
                continue
            reviews_this_page += 1
            all_reviews.append({
                "platform": "appstore",
                "author": e.get("author", {}).get("name", {}).get("label", ""),
                "title": e.get("title", {}).get("label", ""),
                "content": e.get("content", {}).get("label", ""),
                "rating": int(e["im:rating"]["label"]),
                "version": e.get("im:version", {}).get("label", ""),
                "vote_sum": int(e.get("im:voteSum", {}).get("label", "0")),
                "vote_count": int(e.get("im:voteCount", {}).get("label", "0")),
                "updated": e.get("updated", {}).get("label", ""),
                "review_id": e.get("id", {}).get("label", ""),
            })

        print(f"[AppStore] page {page} 수집 리뷰 수: {reviews_this_page}")
        # 서버에 부담 안 주도록 딜레이
        time.sleep(sleep_sec)

        # 혹시 이 페이지에서 리뷰가 0개면 바로 종료
        if reviews_this_page == 0:
            break

    df = pd.DataFrame(all_reviews)
    print(f"[AppStore] 총 수집 리뷰 수: {len(df)}")
    return df


# ===============================
# 3. 구글플레이 리뷰 수집 함수
# ===============================

def fetch_google_play_reviews(app_id: str,
                              lang: str = "ko",
                              country: str = "kr") -> pd.DataFrame:
    """
    google_play_scraper의 reviews_all로 모든 리뷰 가져오기
    - app_id: 패키지명 (예: 'com.voyagerx.vrew.android')
    - lang: 리뷰 언어
    - country: 스토어 국가
    """
    result = reviews_all(
        app_id,
        lang=lang,
        country=country,
        sleep_milliseconds=0,
    )

    df = pd.DataFrame(result)
    if not df.empty:
        # 통일을 위해 컬럼명 일부 맞추기
        df.rename(columns={
            "userName": "author",
            "content": "content",
            "score": "rating",
        }, inplace=True)
        df["platform"] = "googleplay"

    print(f"[GooglePlay] 총 수집 리뷰 수: {len(df)}")
    return df


# ===============================
# 4. 메인 실행 함수
# ===============================

def main():
    # 1) 크롤링할 앱 URL 설정 (여기에 다른 앱 URL 넣어도 됨)
    appstore_url = "https://apps.apple.com/kr/app/vrew-%EB%B8%8C%EB%A3%A8/id1477811799"
    gplay_url = "https://play.google.com/store/apps/details?id=com.voyagerx.vrew.android"

    # 2) URL에서 ID 추출
    appstore_id = get_appstore_id_from_url(appstore_url)
    gplay_id = get_gplay_id_from_url(gplay_url)

    print("=== ID 추출 결과 ===")
    print("App Store ID :", appstore_id)
    print("Google Play ID:", gplay_id)
    print("===================")

    # 3) 리뷰 수집
    appstore_df = fetch_app_store_reviews(appstore_id, country="kr", pages=10, sleep_sec=1.0)
    gplay_df = fetch_google_play_reviews(gplay_id, lang="ko", country="kr")

    # 4) CSV 개별 저장
    if not appstore_df.empty:
        appstore_df.to_csv("vrew_appstore_reviews.csv", index=False, encoding="utf-8-sig")
        print("[SAVE] vrew_appstore_reviews.csv 저장 완료")

    if not gplay_df.empty:
        gplay_df.to_csv("vrew_googleplay_reviews.csv", index=False, encoding="utf-8-sig")
        print("[SAVE] vrew_googleplay_reviews.csv 저장 완료")

    # 5) 통합 데이터프레임 + CSV
    if not appstore_df.empty or not gplay_df.empty:
        combined_df = pd.concat([appstore_df, gplay_df], ignore_index=True, sort=False)
        combined_df.to_csv("vrew_reviews_combined.csv", index=False, encoding="utf-8-sig")
        print("[SAVE] vrew_reviews_combined.csv 저장 완료")
    else:
        print("[WARN] 수집된 리뷰가 없습니다. 통합 CSV는 생성하지 않습니다.")


if __name__ == "__main__":
    main()
