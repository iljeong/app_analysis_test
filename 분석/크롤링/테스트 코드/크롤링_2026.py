# -*- coding: utf-8 -*-
"""
Vrew 리뷰 재크롤링 (2026년 3월)
- App Store: kr, us, jp (RSS 전체 페이지)
- Google Play: kr, us, jp (continuation_token 활용)
- 저장 경로: CSV 데이터/ 폴더
"""

import requests
import pandas as pd
import time
from urllib.parse import urlparse, parse_qs
from google_play_scraper import reviews, Sort

# 저장 경로
SAVE_DIR = "/Users/seojeong-il/Desktop/내문서/데이터 분석/개인 분석/보이저엑스/vrew/CSV 데이터"


def get_appstore_id_from_url(url: str) -> str:
    path_parts = urlparse(url).path.split("/")
    for part in reversed(path_parts):
        if part.startswith("id") and part[2:].isdigit():
            return part[2:]
    for part in reversed(path_parts):
        if part.isdigit():
            return part
    raise ValueError("App Store ID를 URL에서 찾을 수 없습니다.")


def get_gplay_id_from_url(url: str) -> str:
    qs = parse_qs(urlparse(url).query)
    if "id" in qs:
        return qs["id"][0]
    raise ValueError("Google Play app id를 URL에서 찾을 수 없습니다.")


def fetch_app_store_reviews(app_id: str,
                            country: str = "kr",
                            max_pages: int = 1000,
                            sleep_sec: float = 1.0) -> pd.DataFrame:
    all_reviews = []
    consecutive_empty_pages = 0
    max_consecutive_empty = 3

    print(f"[AppStore] 리뷰 수집 시작 (country={country})")

    for page in range(1, max_pages + 1):
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"page={page}/id={app_id}/sortby=mostrecent/json"
        )

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    break
                continue

            data = resp.json()

            if "feed" not in data or "entry" not in data["feed"]:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    break
                continue

            entries = data["feed"]["entry"]
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
                    "country": country,
                })

            if reviews_this_page > 0:
                consecutive_empty_pages = 0
                print(f"  page {page}: {reviews_this_page}개 (누적: {len(all_reviews)}개)")
            else:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    break

            time.sleep(sleep_sec)

        except Exception as e:
            print(f"  page {page} 에러: {e}")
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= max_consecutive_empty:
                break

    df = pd.DataFrame(all_reviews)
    print(f"[AppStore] {country} 총 수집: {len(df)}개")
    return df


def fetch_google_play_reviews(app_id: str,
                              lang: str = "ko",
                              country: str = "kr",
                              count_per_request: int = 200) -> pd.DataFrame:
    all_reviews = []
    continuation_token = None
    request_count = 0

    print(f"[GooglePlay] 리뷰 수집 시작 (lang={lang}, country={country})")

    while True:
        request_count += 1

        try:
            result, continuation_token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=Sort.NEWEST,
                count=count_per_request,
                continuation_token=continuation_token
            )

            if not result:
                break

            all_reviews.extend(result)
            print(f"  요청 {request_count}: {len(result)}개 (누적: {len(all_reviews)}개)")

            if continuation_token is None:
                break

            time.sleep(0.5)

        except Exception as e:
            print(f"  요청 {request_count} 에러: {e}")
            break

    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df.rename(columns={
            "userName": "author",
            "content": "content",
            "score": "rating",
        }, inplace=True)
        df["platform"] = "googleplay"
        df["lang"] = lang
        df["country"] = country

        if "reviewId" in df.columns:
            df.drop_duplicates(subset=["reviewId"], inplace=True)

    print(f"[GooglePlay] {lang}/{country} 총 수집: {len(df)}개")
    return df


def main():
    appstore_url = "https://apps.apple.com/kr/app/vrew-%EB%B8%8C%EB%A3%A8/id1477811799"
    gplay_url = "https://play.google.com/store/apps/details?id=com.voyagerx.vrew.android"

    appstore_id = get_appstore_id_from_url(appstore_url)
    gplay_id = get_gplay_id_from_url(gplay_url)

    print("=" * 50)
    print(f"App Store ID : {appstore_id}")
    print(f"Google Play ID: {gplay_id}")
    print("=" * 50)

    # App Store: kr, us, jp
    appstore_countries = ["kr", "us", "jp"]
    appstore_frames = []
    for country in appstore_countries:
        df = fetch_app_store_reviews(appstore_id, country=country, max_pages=1000, sleep_sec=1.0)
        if not df.empty:
            appstore_frames.append(df)
        print()

    appstore_df = pd.concat(appstore_frames, ignore_index=True, sort=False) if appstore_frames else pd.DataFrame()

    # Google Play: kr, us, jp
    gplay_locales = [
        {"lang": "ko", "country": "kr"},
        {"lang": "en", "country": "us"},
        {"lang": "ja", "country": "jp"},
    ]
    gplay_frames = []
    for locale in gplay_locales:
        df = fetch_google_play_reviews(gplay_id, lang=locale["lang"], country=locale["country"])
        if not df.empty:
            gplay_frames.append(df)
        print()

    gplay_df = pd.concat(gplay_frames, ignore_index=True, sort=False) if gplay_frames else pd.DataFrame()

    # 저장
    print("=" * 50)

    if not appstore_df.empty:
        path = f"{SAVE_DIR}/vrew_appstore_reviews_2026.csv"
        appstore_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] App Store: {len(appstore_df)}개 -> {path}")

    if not gplay_df.empty:
        path = f"{SAVE_DIR}/vrew_googleplay_reviews_2026.csv"
        gplay_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] Google Play: {len(gplay_df)}개 -> {path}")

    if not appstore_df.empty or not gplay_df.empty:
        combined_df = pd.concat([appstore_df, gplay_df], ignore_index=True, sort=False)
        # 중복 제거
        dedupe_keys = [col for col in ["review_id", "reviewId", "author", "content"] if col in combined_df.columns]
        if dedupe_keys:
            combined_df.drop_duplicates(subset=dedupe_keys + ["platform", "country"], inplace=True)
        path = f"{SAVE_DIR}/vrew_reviews_combined_2026.csv"
        combined_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] 통합: {len(combined_df)}개 -> {path}")

    print("=" * 50)
    print("완료!")


if __name__ == "__main__":
    main()
