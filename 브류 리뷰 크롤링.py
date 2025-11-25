# -*- coding: utf-8 -*-
"""
앱스토어(Apple) + 구글플레이 리뷰 전체 크롤러 (개선 버전)
- 입력: 각 마켓의 앱 URL
- 처리:
    1) URL에서 앱 ID 자동 추출
    2) 앱스토어 RSS 리뷰 전체 수집 (페이지네이션 끝까지)
    3) 구글플레이 reviews_all로 전체 리뷰 수집 (continuation_token 활용)
    4) 각각 CSV 저장 + 통합 CSV 저장
"""

import requests
import pandas as pd
import time
from urllib.parse import urlparse, parse_qs
from google_play_scraper import reviews, Sort


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
# 2. 앱스토어 리뷰 전체 수집 함수 (개선)
# ===============================

def fetch_app_store_reviews(app_id: str,
                            country: str = "kr",
                            max_pages: int = 1000,
                            sleep_sec: float = 1.0) -> pd.DataFrame:
    """
    Apple App Store RSS를 이용해 모든 리뷰 가져오기 (페이지네이션 끝까지)
    - app_id: 숫자 ID (예: '1477811799')
    - country: 스토어 국가 코드 (kr, us 등)
    - max_pages: 최대 페이지 수 (기본 1000, 충분히 큰 값)
    - sleep_sec: 페이지 간 대기 시간
    """
    all_reviews = []
    consecutive_empty_pages = 0
    max_consecutive_empty = 3  # 연속 3페이지 비어있으면 종료
    
    print(f"[AppStore] 리뷰 수집 시작 (app_id={app_id}, country={country})")

    for page in range(1, max_pages + 1):
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"page={page}/id={app_id}/sortby=mostrecent/json"
        )
        
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                print(f"[AppStore] page {page} 요청 실패(status={resp.status_code})")
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= max_consecutive_empty:
                    print(f"[AppStore] 연속 {max_consecutive_empty}회 실패, 수집 종료")
                    break
                continue

            data = resp.json()
            
            # 리뷰가 없거나 마지막 페이지면 종료
            if "feed" not in data or "entry" not in data["feed"]:
                consecutive_empty_pages += 1
                print(f"[AppStore] page {page}에 entry 없음 (연속 {consecutive_empty_pages}회)")
                if consecutive_empty_pages >= max_consecutive_empty:
                    print(f"[AppStore] 더 이상 리뷰 없음, 수집 종료")
                    break
                continue

            entries = data["feed"]["entry"]
            reviews_this_page = 0
            
            # 첫 entry는 앱 메타정보인 경우가 많아서 rating 없는 것 제외
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

            if reviews_this_page > 0:
                consecutive_empty_pages = 0
                print(f"[AppStore] page {page} 수집: {reviews_this_page}개 (누적: {len(all_reviews)}개)")
            else:
                consecutive_empty_pages += 1
                print(f"[AppStore] page {page}에 리뷰 없음 (연속 {consecutive_empty_pages}회)")
                if consecutive_empty_pages >= max_consecutive_empty:
                    print(f"[AppStore] 더 이상 리뷰 없음, 수집 종료")
                    break

            # 서버에 부담 안 주도록 딜레이
            time.sleep(sleep_sec)
            
        except Exception as e:
            print(f"[AppStore] page {page} 에러: {e}")
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= max_consecutive_empty:
                break

    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df["country"] = country
    print(f"[AppStore] 총 수집 리뷰 수: {len(df)}")
    return df


# ===============================
# 3. 구글플레이 리뷰 전체 수집 함수 (개선)
# ===============================

def fetch_google_play_reviews(app_id: str,
                              lang: str = "ko",
                              country: str = "kr",
                              count_per_request: int = 200) -> pd.DataFrame:
    """
    google_play_scraper를 이용해 모든 리뷰 가져오기 (continuation_token 활용)
    - app_id: 패키지명 (예: 'com.voyagerx.vrew.android')
    - lang: 리뷰 언어
    - country: 스토어 국가
    - count_per_request: 한 번에 가져올 리뷰 수 (최대 200)
    """
    all_reviews = []
    continuation_token = None
    request_count = 0
    
    print(f"[GooglePlay] 리뷰 수집 시작 (app_id={app_id})")
    
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
                print(f"[GooglePlay] 더 이상 리뷰 없음, 수집 종료")
                break
            
            all_reviews.extend(result)
            print(f"[GooglePlay] 요청 {request_count}: {len(result)}개 수집 (누적: {len(all_reviews)}개)")
            
            # continuation_token이 없으면 마지막 페이지
            if continuation_token is None:
                print(f"[GooglePlay] 마지막 페이지 도달, 수집 종료")
                break
            
            # 서버 부담 방지를 위한 딜레이
            time.sleep(0.5)
            
        except Exception as e:
            print(f"[GooglePlay] 요청 {request_count} 에러: {e}")
            # 에러 발생 시에도 지금까지 수집한 리뷰는 반환
            break
    
    df = pd.DataFrame(all_reviews)
    if not df.empty:
        # 통일을 위해 컬럼명 일부 맞추기
        df.rename(columns={
            "userName": "author",
            "content": "content",
            "score": "rating",
        }, inplace=True)
        df["platform"] = "googleplay"
        df["lang"] = lang
        df["country"] = country
        
        # 중복 제거 (혹시 모를 중복 방지)
        if "reviewId" in df.columns:
            df.drop_duplicates(subset=["reviewId"], inplace=True)
            print(f"[GooglePlay] 중복 제거 후: {len(df)}개")

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
    try:
        appstore_id = get_appstore_id_from_url(appstore_url)
        gplay_id = get_gplay_id_from_url(gplay_url)

        print("=" * 50)
        print("=== ID 추출 결과 ===")
        print(f"App Store ID : {appstore_id}")
        print(f"Google Play ID: {gplay_id}")
        print("=" * 50)
        print()
    except Exception as e:
        print(f"[ERROR] ID 추출 실패: {e}")
        return

    # 3) 리뷰 수집
    appstore_countries = ["kr", "us", "jp"]
    gplay_locales = [
        {"lang": "ko", "country": "kr"},
        {"lang": "en", "country": "us"},
        {"lang": "ja", "country": "jp"},
    ]

    appstore_frames = []
    for country in appstore_countries:
        df = fetch_app_store_reviews(
            appstore_id,
            country=country,
            max_pages=1000,
            sleep_sec=1.0,
        )
        if not df.empty:
            appstore_frames.append(df)
        print()
    appstore_df = pd.concat(appstore_frames, ignore_index=True, sort=False) if appstore_frames else pd.DataFrame()

    gplay_frames = []
    for locale in gplay_locales:
        df = fetch_google_play_reviews(
            gplay_id,
            lang=locale["lang"],
            country=locale["country"],
            count_per_request=200,
        )
        if not df.empty:
            gplay_frames.append(df)
        print()
    gplay_df = pd.concat(gplay_frames, ignore_index=True, sort=False) if gplay_frames else pd.DataFrame()

    print()
    print("=" * 50)
    
    # 4) CSV 개별 저장
    if not appstore_df.empty:
        appstore_df.to_csv("vrew_appstore_reviews.csv", index=False, encoding="utf-8-sig")
        print(f"[SAVE] vrew_appstore_reviews.csv 저장 완료 ({len(appstore_df)}개)")
    else:
        print("[WARN] 앱스토어 리뷰가 없습니다.")

    if not gplay_df.empty:
        gplay_df.to_csv("vrew_googleplay_reviews.csv", index=False, encoding="utf-8-sig")
        print(f"[SAVE] vrew_googleplay_reviews.csv 저장 완료 ({len(gplay_df)}개)")
    else:
        print("[WARN] 구글플레이 리뷰가 없습니다.")

    # 5) 통합 데이터프레임 + CSV
    if not appstore_df.empty or not gplay_df.empty:
        combined_df = pd.concat([appstore_df, gplay_df], ignore_index=True, sort=False)
        dedupe_keys = [col for col in ["review_id", "reviewId", "author", "content"] if col in combined_df.columns]
        if dedupe_keys:
            combined_df.drop_duplicates(subset=dedupe_keys + ["platform", "country"], inplace=True)
        combined_df.to_csv("vrew_reviews_combined.csv", index=False, encoding="utf-8-sig")
        print(f"[SAVE] vrew_reviews_combined.csv 저장 완료 (총 {len(combined_df)}개)")
    else:
        print("[WARN] 수집된 리뷰가 없습니다. 통합 CSV는 생성하지 않습니다.")
    
    print("=" * 50)
    print("✅ 모든 작업 완료!")


if __name__ == "__main__":
    main()